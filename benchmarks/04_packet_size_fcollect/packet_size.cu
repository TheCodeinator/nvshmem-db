#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include <cuda.h>
#include <c++/10/array>
#include "nvshmem.h"
#include "Meas.cuh"

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

constexpr long long MAX_SEND_SIZE{1024 * 1024 * 16};

consteval size_t log2const(size_t n) {
    return n == 1 ? 0 : 1 + log2const(n >> 1);
}

// TODO: verify results make sense and benchmark code is bug-free
// TODO: print in csv format

// from 2 go up to the max packet size in exponential steps
constexpr size_t N_TESTS{log2const(MAX_SEND_SIZE) + 1};

enum SendState {
    RUNNING = 0,
    FINISHED = 1,
};

// list of message sizes to test
// send with nbi pu tin loop with configurable iterations
// nvshmem_quiet afterwards
// record time for each message size

__global__ void exchange_data(int this_pe,
                              uint8_t *const data_src,
                              uint8_t *const data_dest,
                              uint32_t *const flag,
                              const uint64_t n_iterations,
                              Meas *const meas_dev) {
    const int other_pe = static_cast<int>(!this_pe); // there are two PEs in total
    const uint32_t thread_global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t thread_stride = blockDim.x * gridDim.y;


    for (size_t test{0}; test < N_TESTS; ++test) {
        // sync with other PE to make them start simultaneously
        nvshmem_barrier_all();

        meas_dev[test].start = clock64();

        for (size_t i{0}; i < n_iterations; ++i) {
            nvshmem_uint8_fcollect(NVSHMEM_TEAM_WORLD, data_dest, data_src, pow(2, test));
        }

        meas_dev[test].stop = clock64();
    }

}

// TODO: use host to measure time since GPU clock frq. can change dynamically and is therefore not reliable

/**
 * cmd arguments:
 * 0) program name (implicit)
 * 1) number of iterations
 */
int main(int argc, char *argv[]) {
    // init nvshmem
    int n_pes, this_pe;
    cudaStream_t stream;

    assert(argc == 2);
    const u_int64_t n_iterations = std::stoull(argv[1]);
//    const u_int32_t grid_dim = stoi(argv[2]);
//    const u_int32_t block_dim = stoi(argv[3]);
    const u_int32_t grid_dim = 1;
    const u_int32_t block_dim = 1;

    nvshmem_init();
    this_pe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    n_pes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    cudaSetDevice(this_pe);
    cudaStreamCreate(&stream);

    // this test is supposed to be executed on 2 PEs, each sends and receives data from the other PE
    assert(n_pes == 2);

    // allocate symmetric device memory for sending/receiving the data
    auto *const data_src = static_cast<uint8_t *>(nvshmem_malloc(MAX_SEND_SIZE));
    auto *const data_dest = static_cast<uint8_t *>(nvshmem_malloc(MAX_SEND_SIZE * n_pes));
    auto *const flag = static_cast<int *>(nvshmem_malloc(sizeof(uint32_t)));

    // memory for storing the measurements and returning them from device to host
    Meas meas_host[N_TESTS];
    Meas *meas_dev;
    cudaMalloc(&meas_dev, sizeof(Meas) * N_TESTS);

    // call benchmarking kernel
    void *args[] = {&this_pe,
                    const_cast<uint8_t **>(&data_src),
                    const_cast<uint8_t **>(&data_dest),
                    const_cast<int **>(&flag),
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
    nvshmem_free(data_src);
    nvshmem_free(data_dest);
    nvshmem_free(flag);

    for (size_t i{0}; i < N_TESTS; ++i) {
        usleep(this_pe * N_TESTS + i * 100);
        // have send 2^i bytes in each iteation
        std::cout << (this_pe == 0 ? "send" : "receive") << "_" << i << "(" << pow(2, i) << "B)" << ": "
                  << meas_host[i].to_string(n_iterations * pow(2, i)) << std::endl;
    }

    // TODO: print results in suitable CSV format

//    std::ofstream outfile;
//    outfile.open("results.csv");
//    outfile << "type, node_count,in n,out n" << std::endl;
//
//    outfile.close();
}
