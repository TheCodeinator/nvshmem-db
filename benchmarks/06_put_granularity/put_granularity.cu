#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include <cuda.h>
#include <vector>
#include "nvshmem.h"
#include "NVSHMEMUtils.cuh"

constexpr long long MAX_SEND_SIZE{1024 * 1024 * 1024};
constexpr size_t N_TESTS{log2const(MAX_SEND_SIZE) + 1};

__global__ void generalized_benchmark(uint8_t *data_source,
                                      uint8_t *data_sink,
                                      const int this_pe,
                                      const uint64_t num_bytes,
                                      const uint32_t size_per_element) {
    if (this_pe != 0) {
        nvshmem_barrier_all();
        return;
    }
    const uint32_t thread_id = global_thread_id();
    const uint32_t size_per_thread = size_per_element / global_thread_count();
    const uint32_t thread_offset = thread_id * size_per_thread;

    // data_size 1x10^9
    // num_bytes 512
    // size_per_element 64

    // print all variables in one line
//    printf("thread_id %u, thread_offset %u, size_per_thread %u, size_per_element %u, num_bytes %lu\n",
//           thread_id, thread_offset, size_per_thread, size_per_element, num_bytes);

    for (size_t i = 0; i < num_bytes / size_per_element; i += size_per_element) {
        assert(i * size_per_element + thread_offset < MAX_SEND_SIZE);
        nvshmem_uint8_put_nbi(
                data_sink + i * size_per_element + thread_offset,
                data_source + i * size_per_element + thread_offset,
                size_per_element,
                1);
    }

    nvshmem_quiet();
    nvshmem_barrier_all();
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <grid_dims> <block_dims> <num_hosts> <size_per_element>"
                  << std::endl;
        return 1;
    }

    int n_pes, this_pe;
    cudaStream_t stream;

    const uint32_t grid_dim = std::stoi(argv[1]);
    const uint32_t block_dim = std::stoi(argv[2]);
    const uint32_t size_per_element = std::stoi(argv[4]);

    nvshmem_init();
    this_pe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    n_pes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    cudaSetDevice(this_pe);
    cudaStreamCreate(&stream);

    if (n_pes != 2) {
        std::cerr << "This test has to be started with exactly 2 PEs." << std::endl;
        return 1;
    }

    auto data_size = MAX_SEND_SIZE * block_dim * grid_dim;
    size_t *data_source;
    cudaMalloc(&data_source, data_size);
    auto *data_sink = static_cast<uint8_t *>(nvshmem_malloc(data_size));

    for (size_t test = 0; test < N_TESTS; ++test) {
        const auto msg_size = int_pow(2, test);
        const auto time_taken = time_kernel(generalized_benchmark, grid_dim, block_dim, 0, stream,
                                            data_source, data_sink, this_pe, msg_size, size_per_element);
        nvshmem_quiet();
        if (this_pe == 0) {
            std::cout << msg_size << " bytes took " << time_taken.count() << " microseconds." << std::endl;
        }
    }

    cudaFree(data_source);
    nvshmem_free(data_sink);
    nvshmem_finalize();
    return 0;
}
