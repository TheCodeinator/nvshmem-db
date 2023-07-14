#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include <cuda.h>
#include "nvshmem.h"

// used to check the status code of cuda routines for errors
#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t _CHECK_result = (stmt);                                              \
        if (cudaSuccess != _CHECK_result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(_CHECK_result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

// used to check the status code of NVSHMEM routines for errors
#define NVSHMEM_CHECK(stmt)                                                                \
    do {                                                                                   \
        int _CHECK_result = (stmt);                                                               \
        if (NVSHMEMX_SUCCESS != _CHECK_result) {                                                  \
            fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, \
                    _CHECK_result);                                                               \
            exit(-1);                                                                      \
        }                                                                                  \
    } while (0)


constexpr long long SHADER_FREQ_KHZ{1530000};

struct TimeMeas {
    long long start = 0;
    long long stop = 0;

    __host__ __device__ [[nodiscard]] inline long long diff() const {
        return stop - start;
    }

    __host__ __device__ [[nodiscard]] double diff_ms() const {
        // NOTE: long double type not supported in device code (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html: 14.5.15)
        return static_cast<double>(this->diff()) / SHADER_FREQ_KHZ;
    }
};

// TODO: verify results make sense
// TODO: return results to CPU and print in csv format


enum SendState {
    RUNNING = 0,
    FINISHED = 1,
};

__device__ TimeMeas send_one_thread_sep(uint8_t *const data,
                                        const int other_pe,
                                        uint32_t *const flag,
                                        const uint32_t n_elems,
                                        const uint32_t n_iterations) {
    TimeMeas time{};

    // sync with other PE to make them start simultaneously
    nvshmem_barrier_all();

    time.start = clock64();

    // send data to other PE at same position
    for (size_t it{0}; it < n_iterations; ++it) {
        for (size_t i{0}; i < n_elems; ++i) {
            nvshmem_uint8_put_nbi(data + i,
                                  data + i,
                                  1,
                                  other_pe);
        }
    }

    // let following mem operations be executed after the previous sending
    nvshmem_fence();

    // set memory flag at other PE to signal that all previous send operation must have been completed (see fence)
    nvshmem_uint32_put_nbi(flag, flag, 1, other_pe); // send flag

    // make sure all send buffers are reusable
    nvshmem_quiet();

    time.stop = clock64();

    return time;
}


__device__ TimeMeas send_one_thread_once(uint8_t *const data,
                                         const int other_pe,
                                         uint32_t *const flag,
                                         const uint32_t n_elems,
                                         const uint32_t n_iterations) {
    TimeMeas time{};

    // synchronize with other PE to make them start next test simultaneously
    nvshmem_barrier_all();

    time.start = clock64();

    // send data in one go
    for (size_t it{0}; it < n_iterations; ++it) {
        nvshmem_uint8_put_nbi(data,
                              data,
                              n_elems,
                              other_pe);
    }

    // let following mem operations be executed after the previous sending
    nvshmem_fence();

    // set memory flag at other PE to signal that all previous send operation must have been completed (see fence)
    nvshmem_uint32_put_nbi(flag, flag, 1, other_pe);

    // make sure all send buffers are reusable
    nvshmem_quiet();

    time.stop = clock64();

    return time;
}

__device__ TimeMeas send_multi_thread_sep(uint8_t *const data,
                                          const int other_pe,
                                          uint32_t *const flag,
                                          const uint32_t n_elems,
                                          const uint32_t n_iterations) {
    const uint32_t thread_global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t thread_stride = blockDim.x * gridDim.y;
    TimeMeas time{};

    // sync with other PE to make them start simultaneously
    if (thread_global_id == 0) {
        nvshmem_barrier_all();
    }

    if (thread_global_id == 0) {
        time.start = clock64();
    }

    // start for loop together
    __syncthreads();

    // send data to other PE at same position
    for (size_t it{0}; it < n_iterations; ++it) {
        for (size_t i{thread_global_id}; i < n_elems; i += thread_stride) {
            nvshmem_uint8_put_nbi(data + i,
                                  data + i,
                                  1,
                                  other_pe);
        }
    }

    // let following mem operations be executed after the previous sending
    nvshmem_fence();

    // set memory flag at other PE to signal that all previous send operation must have been completed (see fence)
    if (thread_global_id == 0) {
        nvshmem_uint32_put_nbi(flag, flag, 1, other_pe); // send flag
    }

    // make sure all send buffers are reusable
    nvshmem_quiet();

    if (thread_global_id == 0) {
        time.stop = clock64();
    }

    return time;
}

__device__ TimeMeas time_recv(uint8_t *const data,
                              const int other_pe,
                              volatile uint32_t *const flag,
                              const uint32_t n_elems) {
    TimeMeas time{};

    // sync with other PE to make them start simultaneously
    nvshmem_barrier_all();

    time.start = clock64();

    // wait until flag has been delivered, this then indicates all previous data has been delivered
    while (*flag == SendState::RUNNING);

    time.stop = clock64();

    // verify correctness
    for (size_t i{0}; i < n_elems; ++i) {
        // write lower bits of index to every element
        assert(data[i] == static_cast<uint8_t>(i));
    }

    // reset receive buffer and flag for next test
    memset(data, 0, n_elems);
    *flag = SendState::RUNNING;

    return time;
}


__global__ void exchange_data(int this_pe,
                              uint8_t *const data,
                              uint32_t *const flag,
                              const uint32_t n_elems,
                              const uint32_t n_iterations) {
    const int other_pe = static_cast<int>(!this_pe); // there are two PEs in total
    const uint32_t thread_global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t thread_stride = blockDim.x * gridDim.y;

    // PE 0 is the sender
    if (this_pe == 0) {
        // populate data to send to PE 1
        for (size_t i{thread_global_id}; i < n_elems; i += thread_stride) {
            // write lower bits of index to every element
            data[i] = static_cast<uint8_t>(i);
        }

        // set local flag to finished state on this PE, send flag every time the sender is finished
        // Receiver will reset its local instance of the flag after each of the tests
        *flag = SendState::FINISHED;

        if (thread_global_id == 0) {
            TimeMeas time = send_one_thread_sep(data, other_pe, flag, n_elems, n_iterations);
            printf("send_one_thread_sep: elems=%u, iterations=%u, start=%lld, stop=%lld time=%lld (%fms)\n",
                   n_elems, n_iterations, time.start, time.stop, time.diff(), time.diff_ms());
        }

        if (thread_global_id == 0) {
            TimeMeas time = send_one_thread_once(data, other_pe, flag, n_elems, n_iterations);
            printf("send_one_thread_once: elems=%u, iterations=%u, start=%lld, stop=%lld time=%lld (%fms)\n",
                   n_elems, n_iterations, time.start, time.stop, time.diff(), time.diff_ms());
        }

        {
            TimeMeas time = send_multi_thread_sep(data, other_pe, flag, n_elems, n_iterations);
            if (thread_global_id == 0) {
                printf("send_multi_thread_sep: elems=%u, iterations=%u, start=%lld, stop=%lld time=%lld (%fms)\n",
                       n_elems, n_iterations, time.start, time.stop, time.diff(), time.diff_ms());
            }
        }

    } else { // PE 1 is the receiver
        // make reads from flag volatile
        volatile uint32_t *flag_vol = flag;

        // receiver does not do anything but waiting, only needs one thread in all scenarios

        constexpr size_t n_tests{3};
        const char *test_names[] = {"recv(send_one_thread_sep)",
                                    "recv(send_one_thread_once)",
                                    "recv(send_multi_thread_sep"};

        if (thread_global_id == 0) {
            for (size_t i{0}; i < n_tests; ++i) {
                TimeMeas time = time_recv(data, other_pe, flag_vol, n_elems);
                printf("%s: elems=%u, iterations=%u, start=%lld, stop=%lld time=%lld (%fms)\n",
                       test_names[i], n_elems, n_iterations, time.start, time.stop, time.diff(), time.diff_ms());
            }
        }
    }
}

/**
 * cmd arguments:
 * 0) program name (implicit)
 * 1) number of elements
 * 2) number of iterations
 * 3) grid dims
 * 4) block dims
 */
int main(int argc, char *argv[]) {
    // init nvshmem
    int n_pes, this_pe;
    cudaStream_t stream;

    assert(argc == 5);
    const u_int32_t n_elems = stoi(argv[1]);
    const u_int32_t n_iterations = stoi(argv[2]);
    const u_int32_t grid_dim = stoi(argv[3]);
    const u_int32_t block_dim = stoi(argv[4]);

    nvshmem_init();
    this_pe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    n_pes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    cudaSetDevice(this_pe);
    cudaStreamCreate(&stream);

    // this test is supposed to be executed on 2 PEs, each sends and receives data from the other PE
    assert(n_pes == 2);

    // allocate symmetric device memory for sending/receiving the data
    auto *const data = static_cast<uint8_t *>(nvshmem_malloc(n_elems));
    auto *const flag = static_cast<int *>(nvshmem_malloc(sizeof(uint32_t)));

    // call benchmarking kernel
    void *args[] = {&this_pe,
                    const_cast<uint8_t **>(&data),
                    const_cast<int **>(&flag),
                    const_cast<uint32_t *>(&n_elems),
                    const_cast<uint32_t *>(&n_iterations)};
    NVSHMEM_CHECK(
            nvshmemx_collective_launch((const void *) exchange_data, grid_dim, block_dim, args, 1024 * 4, stream));

    // wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());


    // TODO: print results in suitable CSV format

//    std::ofstream outfile;
//    outfile.open("results.csv");
//    outfile << "type, node_count,in n,out n" << std::endl;
//
//    outfile.close();
}
