#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include <cuda.h>
#include <vector>
#include <bit>
#include "nvshmem.h"
#include "NVSHMEMUtils.cuh"


__global__ void generalized_benchmark(uint8_t *data_source,
                                      uint8_t *data_sink,
                                      const int this_pe,
                                      const uint64_t num_bytes,
                                      const uint64_t message_size) {
    const uint64_t thread_id = global_thread_id();
    const uint64_t thread_count = global_thread_count();
    const uint64_t thread_offset = thread_id * message_size;

    if (this_pe != 0 && thread_id == 0) {
        // wait to receive data from PE 0
        nvshmem_barrier_all();
        return;
    }

    for (uint64_t i = 0; i < num_bytes / (message_size * thread_count); ++i) {
        nvshmem_uint8_put_nbi(
                data_sink + thread_offset,
                data_source + thread_offset,
                message_size,
                1);
    }

    if (thread_id == 0) {
        nvshmem_quiet();
        // synchronize all PEs (notify PE 1 that data is finished)
        nvshmem_barrier_all();
    }
}

// Do a barrier operation to prevent compile from optimizing out empty kernel
__global__ void warmup() {

    const uint32_t thread_global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_global_id == 0) {
        nvshmem_barrier_all();
    }

}

int main(int argc, char *argv[]) {
    if (argc != 6 && argc != 7) {
        throw std::invalid_argument(
                "Usage: " + std::string(argv[0]) +
                " <grid_dims> <block_dims> <num_hosts> <num_bytes> <max_send_size> [<min_send_size>]");
    }

    int n_pes, this_pe;
    cudaStream_t stream;

    const uint32_t grid_dim = std::stoi(argv[1]);
    const uint32_t block_dim = std::stoi(argv[2]);
    const uint32_t num_hosts = std::stoi(argv[3]);

    // the number of bytes that are sent in total per kernel
    const uint64_t num_bytes = std::stoi(argv[4]);

    // the maximum number of bytes that are sent with a single nvshmem put call (increases in powers of 2 starting at 1)
    const uint64_t max_send_size = std::stoi(argv[5]);

    const uint64_t min_send_size = argc == 7 ? std::stoi(argv[6]) : 1;

    if (min_send_size > max_send_size) {
        throw std::invalid_argument("min_send_size must be smaller than max_send_size");
    }

    const uint64_t buffer_size = grid_dim * block_dim * max_send_size;

    if (std::popcount(max_send_size) != 1) {
        throw std::invalid_argument("max_send_size must be a power of 2");
    }

    if (num_bytes / (buffer_size) < 1) {
        throw std::invalid_argument(
                "num_bytes must be greater than grid_dim * block_dim * max_send_size (= " +
                std::to_string(buffer_size) + ")");
    }

    if (num_bytes % (buffer_size) != 0) {
        throw std::invalid_argument(
                "num_bytes must be a multiple of grid_dim * block_dim * max_send_size (= " +
                std::to_string(buffer_size) + ")");
    }

    nvshmem_init();
    this_pe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    n_pes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    cudaSetDevice(this_pe);
    cudaStreamCreate(&stream);

    if (n_pes != 2) {
        std::cerr << "This test has to be started with exactly 2 PEs." << std::endl;
        return 1;
    }

    // each thread is allocated one MAX_SEND_SIZE element, which is re-sent until num_bytes are sent
    auto *data_source = static_cast<uint8_t *>(nvshmem_malloc(buffer_size));
    auto *data_sink = static_cast<uint8_t *>(nvshmem_malloc(buffer_size));

    // warm up device
    collective_launch(warmup, grid_dim, block_dim, 0, stream);

    for (uint64_t message_size = min_send_size; message_size <= max_send_size; message_size <<= 1) {
        const auto time_taken = time_kernel(generalized_benchmark, grid_dim, block_dim, 0, stream,
                                            data_source, data_sink, this_pe, num_bytes, message_size, max_send_size);
        if (this_pe == 0) {
            std::cout << message_size
                      << " bytes sized packages took " << time_taken.count()
                      << " nanoseconds "
                      << "(" << gb_per_sec(time_taken, num_bytes) << " GB/s)"
                      << std::endl;
        }
    }

    nvshmem_free(data_source);
    nvshmem_free(data_sink);
    nvshmem_finalize();
    return 0;
}
