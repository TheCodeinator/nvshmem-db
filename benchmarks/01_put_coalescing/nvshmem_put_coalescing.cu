#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include <cuda.h>
#include <c++/10/array>
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
constexpr long long SHADER_FREQ_HZ{1530};

struct Meas {
    long long start = 0;
    long long stop = 0;

    __host__  [[nodiscard]] inline long long clock_diff() const {
        return stop - start;
    }

    __host__  [[nodiscard]] long double time_diff_ms() const {
        return static_cast<long double>(this->clock_diff()) / SHADER_FREQ_KHZ;
    }

    __host__  [[nodiscard]] long double time_diff_s() const {
        return static_cast<long double>(this->clock_diff()) / SHADER_FREQ_HZ;
    }

    __host__ [[nodiscard]] long double get_throughput(const uint64_t n_bytes) const {
        return n_bytes / this->time_diff_s() / 1000000;
    }

    __host__ [[nodiscard]] std::string to_csv() const {
        // TODO: @Alex If you want, use this to implement printing to csv format
        return {};
    }

    __host__ std::string to_string(const uint64_t n_bytes) const {
        return "start=" + std::to_string(start) +
               " stop=" + std::to_string(stop) +
               " clock_diff=" + std::to_string(this->clock_diff()) +
               " (" + std::to_string(this->time_diff_ms()) +
               "ms) throughput=" + std::to_string(this->get_throughput(n_bytes)) + " GB/s";
    }
};

// TODO: verify results make sense
// TODO: return results to CPU and print in csv format

constexpr size_t N_TESTS{3};

enum SendState {
    RUNNING = 0,
    FINISHED = 1,
};

__device__ Meas send_one_thread_sep(uint8_t *const data,
                                    const int other_pe,
                                    uint32_t *const flag,
                                    const uint64_t n_elems,
                                    const uint64_t n_iterations) {
    Meas meas{};

    // sync with other PE to make them start simultaneously
    nvshmem_barrier_all();

    meas.start = clock64();

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

    meas.stop = clock64();

    return meas;
}


__device__ Meas send_one_thread_once(uint8_t *const data,
                                     const int other_pe,
                                     uint32_t *const flag,
                                     const uint64_t n_elems,
                                     const uint64_t n_iterations) {
    Meas meas{};

    // synchronize with other PE to make them start next test simultaneously
    nvshmem_barrier_all();

    meas.start = clock64();

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

    meas.stop = clock64();

    return meas;
}

__device__ void send_multi_thread_sep(uint8_t *const data,
                                      const int other_pe,
                                      uint32_t *const flag,
                                      const uint64_t n_elems,
                                      const uint64_t n_iterations,
                                      Meas &meas) {
    const uint32_t thread_global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t thread_stride = blockDim.x * gridDim.y;

    // sync with other PE to make them start simultaneously
    if (thread_global_id == 0) {
        nvshmem_barrier_all();
    }

    if (thread_global_id == 0) {
        meas.start = clock64();
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
        meas.stop = clock64();
    }
}

__device__ Meas time_recv(uint8_t *const data,
                          const int other_pe,
                          volatile uint32_t *const flag,
                          const uint64_t n_elems,
                          const uint64_t n_iterations) {
    Meas meas{};

    // sync with other PE to make them start simultaneously
    nvshmem_barrier_all();

    meas.start = clock64();

    // wait until flag has been delivered, this then indicates all previous data has been delivered
    while (*flag == SendState::RUNNING);

    meas.stop = clock64();

    // verify correctness
    for (size_t i{0}; i < n_elems; ++i) {
        // write lower bits of index to every element
        assert(data[i] == static_cast<uint8_t>(i));
    }

    // reset receive buffer and flag for next test
    memset(data, 0, n_elems);
    *flag = SendState::RUNNING;

    return meas;
}


__global__ void exchange_data(int this_pe,
                              uint8_t *const data,
                              uint32_t *const flag,
                              const uint64_t n_elems,
                              const uint64_t n_iterations,
                              Meas *const meas_dev) {
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
            meas_dev[0] = send_one_thread_sep(data, other_pe, flag, n_elems, n_iterations);
        }

        if (thread_global_id == 0) {
            meas_dev[1] = send_one_thread_once(data, other_pe, flag, n_elems, n_iterations);
        }

        {
            send_multi_thread_sep(data, other_pe, flag, n_elems, n_iterations, meas_dev[2]);
        }

    } else { // PE 1 is the receiver
        // make reads from flag volatile
        volatile uint32_t *flag_vol = flag;

        // receiver does not do anything but waiting, only needs one thread in all scenarios
        if (thread_global_id == 0) {
            for (size_t i{0}; i < N_TESTS; ++i) {
                meas_dev[i] = time_recv(data, other_pe, flag_vol, n_elems, n_iterations);
            }
        }
    }
}

// TODO: use host to measure time since GPU clock frq. can change dynamically and is therefore not reliable

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
    const u_int64_t n_elems = std::stoull(argv[1]);
    const u_int64_t n_iterations = std::stoull(argv[2]);
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

    // memory for storing the measurements and returning them from device to host
    Meas meas_host[N_TESTS];
    Meas *meas_dev;
    cudaMalloc(&meas_dev, sizeof(Meas) * N_TESTS);

    // call benchmarking kernel
    void *args[] = {&this_pe,
                    const_cast<uint8_t **>(&data),
                    const_cast<int **>(&flag),
                    const_cast<uint64_t *>(&n_elems),
                    const_cast<uint64_t *>(&n_iterations),
                    &meas_dev};
    NVSHMEM_CHECK(
            nvshmemx_collective_launch((const void *) exchange_data, grid_dim, block_dim, args, 1024 * 4, stream));

    // wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy results to host
    cudaMemcpy(meas_host, meas_dev, sizeof(Meas) * N_TESTS, cudaMemcpyDeviceToHost);

    // deallocate all the memory that has been alocated
    cudaFree(meas_dev);
    nvshmem_free(data);
    nvshmem_free(flag);

    std::array<std::string, N_TESTS> test_names;

    // print results
    if (this_pe == 0) { // sender
        test_names = {"send_one_thread_sep",
                      "send_one_thread_once",
                      "send_multi_thread_sep"};
    } else { // receiver
        test_names = {"recv(send_one_thread_sep)",
                      "recv(send_one_thread_once)",
                      "recv(send_multi_thread_sep"};
    }

    for (size_t i{0}; i < N_TESTS; ++i) {
        usleep(this_pe * N_TESTS + i * 100);
        std::cout << test_names[i] << ": " << meas_host[i].to_string(n_iterations * n_elems) << std::endl;
    }

    // TODO: print results in suitable CSV format

//    std::ofstream outfile;
//    outfile.open("results.csv");
//    outfile << "type, node_count,in n,out n" << std::endl;
//
//    outfile.close();
}
