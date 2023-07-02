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

constexpr size_t N_ELEMS{64 * 1024 * 1024};

__global__ void exchange_data(int this_pe,
                              uint8_t *const data_send,
                              uint8_t *const data_recv) {
    const int other_pe = static_cast<int>(!this_pe); // there are two PEs in total

    // we only use one thread
    if (threadIdx.x != 0) {
        return;
    }

    // populate data to send to other PE
    for (size_t i{0}; i < N_ELEMS; ++i) {
        // write pe number into every element
        data_send[i] = static_cast<uint8_t >(this_pe);
    }

    // exchange data using individual async put calls
    for (size_t i{0}; i < N_ELEMS; ++i) {
        nvshmem_uint8_put_nbi(data_recv + i,
                              data_send + i,
                              1,
                              other_pe);
    }

    // TODO: continue here
    //  finish first sending, verify result and do second sending
    // TODO: write one kernel that populates our send buffer and then write one kernel for each of the things we test
}


// prepare data
// init nvshmem
// send data via put all with on e operation
// queit and barrier
// send data via put with many operations

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

    // allocate symmetric device memory for sending and receiving
    auto *const data_send = static_cast<uint8_t *>(nvshmem_malloc(N_ELEMS));
    auto *const data_recv = static_cast<uint8_t *>(nvshmem_malloc(N_ELEMS));

    void *args[] = {&this_pe,
                    const_cast<uint8_t **>(&data_send),
                    const_cast<uint8_t **>(&data_send)};

    NVSHMEM_CHECK(nvshmemx_collective_launch((const void *) exchange_data, 1, 1, args, 1024 * 4, stream));


    CUDA_CHECK(cudaMalloc(&offsets, n_pes * sizeof(uint32_t)));


    std::ofstream outfile;
    outfile.open("bench_shuffle_out.csv");
    outfile << "type, node_count,in n,out n" << std::endl;

    for (int tableSize{1000}; tableSize <= 100000; tableSize *= 2) {

        outfile << "nvshmem_shuffle, 1, " << tableSize << ", ";

        auto start = std::chrono::steady_clock::now();
        sleep(1e-4 * tableSize);
        auto end = std::chrono::steady_clock::now();

        auto dur = end - start;
        auto time_ms = dur.count() * 1e-6;

        outfile << time_ms << "\n";

    }

    outfile.close();
}
