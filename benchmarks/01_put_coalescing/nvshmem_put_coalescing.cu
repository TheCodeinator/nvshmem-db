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


constexpr size_t N_ELEMS{1024 * 1024};
constexpr size_t N_ITERATIONS{10};
constexpr long long SHADER_FREQ_KHZ{1530000};

struct TimeMeas {
    long long start = 0;
    long long stop = 0;

    __host__ __device__ [[nodiscard]] inline long long diff() const {
        return stop - start;
    }

    __host__ __device__ [[nodiscard]] double diff_ms() {
        // NOTE: long double type not supported in device code (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html: 14.5.15)
        return static_cast<double>(this->diff()) / SHADER_FREQ_KHZ;
    }
};

// TODO: also print in ms
// TODO: verify results make sense


enum SendState {
    RUNNING = 0,
    FINISHED = 1,
};

__device__ TimeMeas send_one_thread_sep(uint8_t *const data,
                                        const int other_pe,
                                        uint32_t *const flag) {
    TimeMeas time{};

    // sync with other PE to make them start simultaneously
    nvshmem_barrier_all();

    time.start = clock64();

    // send data to other PE at same position
    for (size_t it{0}; it < N_ITERATIONS; ++it) {
        for (size_t i{0}; i < N_ELEMS; ++i) {
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
                                         uint32_t *const flag) {
    TimeMeas time{};

    // synchronize with other PE to make them start next test simultaneously
    nvshmem_barrier_all();

    time.start = clock64();

    // send data in one go
    for (size_t it{0}; it < N_ITERATIONS; ++it) {
        nvshmem_uint8_put_nbi(data,
                              data,
                              N_ELEMS,
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

__device__ TimeMeas time_recv(uint8_t *const data,
                              const int other_pe,
                              volatile uint32_t *const flag) {
    TimeMeas time{};

    // sync with other PE to make them start simultaneously
    nvshmem_barrier_all();

    time.start = clock64();

    // wait until flag has been delivered, this then indicates all previous data has been delivered
    while (*flag == SendState::RUNNING);

//    nvshmemi_wait_until(flag, NVSHMEM_CMP_EQ, SEND_FINISHED);

    time.stop = clock64();

    // verify correctness
    for (size_t i{0}; i < N_ELEMS; ++i) {
        // write lower bits of index to every element
        assert(data[i] == static_cast<uint8_t>(i));
    }

    // reset receive buffer and flag for next test
    memset(data, 0, N_ELEMS);
    *flag = SendState::RUNNING;

    return time;
}


__global__ void exchange_data(int this_pe,
                              uint8_t *const data,
                              uint32_t *const flag) {
    const int other_pe = static_cast<int>(!this_pe); // there are two PEs in total

    // PE 0 is the sender
    if (this_pe == 0) {
        // populate data to send to PE 1
        for (size_t i{0}; i < N_ELEMS; ++i) {
            // write lower bits of index to every element
            data[i] = static_cast<uint8_t >(i);
        }

        // set local flag to finished state on this PE, send flag every time the sender is finished
        // Receiver will reset its local instance of the flag after each of the tests
        *flag = SendState::FINISHED;

        if (threadIdx.x == 0) {
            TimeMeas time = send_one_thread_sep(data, other_pe, flag);
            printf("send_one_thread_sep: elems=%lu, iterations=%lu, start=%lld, stop=%lld time=%lld (%fms)\n",
                   N_ELEMS, N_ITERATIONS, time.start, time.stop, time.diff(), time.diff_ms());
        }

        if (threadIdx.x == 0) {
            TimeMeas time = send_one_thread_once(data, other_pe, flag);
            printf("send_one_thread_once: elems=%lu, iterations=%lu, start=%lld, stop=%lld time=%lld (%fms)\n",
                   N_ELEMS, N_ITERATIONS, time.start, time.stop, time.diff(), time.diff_ms());
        }

    } else { // PE 1 is the receiver
        // make reads from flag volatile
        volatile uint32_t *flag_vol = flag;

        // receiver does not do anything but waiting, only needs one thread in all scenarios

        if (threadIdx.x == 0) {
            TimeMeas time = time_recv(data, other_pe, flag_vol);
            printf("recv(send_one_thread_sep): elems=%lu, iterations=%lu, start=%lld, stop=%lld time=%lld (%fms)\n",
                   N_ELEMS, N_ITERATIONS, time.start, time.stop, time.diff(), time.diff_ms());
        }

        if (threadIdx.x == 0) {
            TimeMeas time = time_recv(data, other_pe, flag_vol);
            printf("recv(send_one_thread_once): elems=%lu, iterations=%lu, start=%lld, stop=%lld time=%lld (%fms)\n",
                   N_ELEMS, N_ITERATIONS, time.start, time.stop, time.diff(), time.diff_ms());
        }
    }
}

int main(int argc, char *argv[]) {
    // init nvshmem
    int n_pes, this_pe;
    cudaStream_t stream;

    nvshmem_init();
    this_pe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    n_pes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    cudaSetDevice(this_pe);
    cudaStreamCreate(&stream);

    // this test is supposed to be executed on 2 PEs, each sends and receives data from the other PE
    assert(n_pes == 2);

    // allocate symmetric device memory for sending/receiving the data
    auto *const data = static_cast<uint8_t *>(nvshmem_malloc(N_ELEMS));
    auto *const flag = static_cast<int *>(nvshmem_malloc(sizeof(uint32_t)));

    // call benchmarking kernel
    void *args[] = {&this_pe,
                    const_cast<uint8_t **>(&data),
                    const_cast<int **>(&flag)};
    NVSHMEM_CHECK(nvshmemx_collective_launch((const void *) exchange_data, 1, 1, args, 1024 * 4, stream));

    // wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());


    // TODO: print results in suitable CSV format

//    std::ofstream outfile;
//    outfile.open("results.csv");
//    outfile << "type, node_count,in n,out n" << std::endl;
//
//    outfile.close();
}
