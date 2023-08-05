#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include <cuda.h>
#include <array>
#include <vector>
#include "nvshmem.h"
#include "NVSHMEMUtils.cuh"
#include "Macros.cuh"

constexpr long long MAX_SEND_SIZE{1024 * 1024};

consteval size_t log2const(size_t n) {
    return n == 1 ? 0 : 1 + log2const(n >> 1);
}

// TODO: verify results make sense and benchmark code is bug-free

// from 2 go up to the max packet size in exponential steps
constexpr size_t N_TESTS{log2const(MAX_SEND_SIZE) + 1};

__global__ void exchange_data(uint8_t *const data_src,
                              uint8_t *const data_dest,
                              const uint64_t n_bytes,
                              const uint32_t msg_size) {
    // send with given msg size until n_elems sent
    for (size_t i{0}; i < (n_bytes / msg_size); ++i) {
        nvshmem_uint8_fcollect(NVSHMEM_TEAM_WORLD, data_dest, data_src, msg_size);
    }

    // make sure all PEs finish together when all have executed every iteration
    nvshmem_barrier_all();
}

/**
 * cmd arguments:
 * 0) program name (implicit)
 * 1) number of bytes to transfer per PE
 */
int main(int argc, char *argv[]) {
    // init nvshmem
    int n_pes, this_pe;
    cudaStream_t stream;

    assert(argc == 2);
    const u_int64_t n_bytes = std::stoull(argv[1]);
    constexpr u_int32_t grid_dim = 1;
    constexpr u_int32_t block_dim = 1;

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

    std::vector<std::pair<uint32_t, std::chrono::microseconds>> measurements{};
    measurements.reserve(N_TESTS);

    for (size_t test{0}; test < N_TESTS; ++test) {
        const uint32_t msg_size = std::pow(2, test);
        measurements.emplace_back(msg_size,
                                  time_kernel(exchange_data, grid_dim, block_dim, 1024 * 4, stream,
                                              data_src, data_dest, n_bytes, msg_size));
    }

    // wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // deallocate all the memory that has been allocated
    nvshmem_free(data_src);
    nvshmem_free(data_dest);

    if (this_pe == 0) {
        for (const auto &meas: measurements) {
            std::cout << "msg_size = " << meas.first << ", throughput = " << gb_per_sec(meas.second, n_bytes) << "GB/s"
                      << std::endl;
        }
    }

    // TODO: print results in suitable CSV format

//    std::ofstream outfile;
//    outfile.open("results.csv");
//    outfile << "type, node_count,in n,out n" << std::endl;
//
//    outfile.close();
}
