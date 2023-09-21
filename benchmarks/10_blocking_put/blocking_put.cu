#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include <cuda.h>
#include <vector>
#include <bit>
#include <iomanip>
#include "nvshmem.h"
#include "NVSHMEMUtils.cuh"

enum KernelMode {
    NBI_QUIET_ALL,
    NBI_QUIET_EACH,
    NBI_NO_QUIET,
    BLOCKING
};


template<KernelMode Mode>

__global__ void kernel(uint8_t *data_source,
                       uint8_t *data_sink,
                       const int this_pe,
                       const uint64_t message_size,
                       const uint64_t n_iterations) {
    const uint64_t thread_id = global_thread_id();

    if (this_pe != 0) {
        // do not sync with other pe, to make it possible to time only sending kernel
        return;
    }

    if constexpr (Mode == NBI_QUIET_ALL) {
        for (uint64_t i = 0; i < n_iterations; ++i) {
            nvshmem_uint8_put_nbi(
                    data_sink,
                    data_source,
                    message_size,
                    1);
        }
        nvshmem_quiet();
    }

    if constexpr (Mode == NBI_QUIET_EACH) {
        for (uint64_t i = 0; i < n_iterations; ++i) {
            nvshmem_uint8_put_nbi(
                    data_sink,
                    data_source,
                    message_size,
                    1);
            nvshmem_quiet();
        }
    }

    if constexpr (Mode == NBI_NO_QUIET) {
        for (uint64_t i = 0; i < n_iterations; ++i) {
            nvshmem_uint8_put_nbi(
                    data_sink,
                    data_source,
                    message_size,
                    1);
        }
    }

    if constexpr (Mode == BLOCKING) {
        for (uint64_t i = 0; i < n_iterations; ++i) {
            nvshmem_uint8_put(
                    data_sink,
                    data_source,
                    message_size,
                    1);
        }
    }

    // return sender without synchronizing with the receiver to make sure we return early i.e. before sending is complete
    // in the case of unawaited async calls
}

// Do a barrier operation to prevent compile from optimizing out empty kernel
__global__ void warmup() {

    const uint32_t thread_global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_global_id == 0) {
        nvshmem_barrier_all();
    }

}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        throw std::invalid_argument("Usage: " + std::string(argv[0]) +
                                    " <num_hosts> <message_size> <n_iterations>");
    }

    const uint64_t num_hosts = std::stoul(argv[1]);
    const uint64_t message_size = std::stoul(argv[2]);
    const uint64_t n_iterations = std::stoul(argv[3]);

    nvshmem_init();
    const int this_pe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    const int n_pes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    cudaSetDevice(this_pe);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    if (n_pes != 2) {
        throw std::invalid_argument("This test has to be started with exactly 2 PEs");
    }

    // each thread is allocated one MAX_SEND_SIZE element, which is re-sent until num_bytes are sent
    auto *data_source = static_cast<uint8_t *>(nvshmem_malloc(message_size));
    auto *data_sink = static_cast<uint8_t *>(nvshmem_malloc(message_size));

    // warm up device
    collective_launch(warmup, 1, 1, 0, stream);


    const auto time_nbi_quiet_all = time_kernel(kernel<NBI_QUIET_ALL>, 1, 1, 0, stream,
                                                data_source, data_sink, this_pe, message_size, n_iterations);

    nvshmem_barrier_all(); // start next kernel together with other PE

    const auto time_nbi_quiet_each = time_kernel(kernel<NBI_QUIET_EACH>, 1, 1, 0, stream,
                                                 data_source, data_sink, this_pe, message_size, n_iterations);
    nvshmem_barrier_all();

    const auto time_nbi_no_quiet = time_kernel(kernel<NBI_NO_QUIET>, 1, 1, 0, stream,
                                               data_source, data_sink, this_pe, message_size, n_iterations);
    nvshmem_barrier_all();

    const auto time_blocking = time_kernel(kernel<BLOCKING>, 1, 1, 0, stream,
                                           data_source, data_sink, this_pe, message_size, n_iterations);

    nvshmem_barrier_all();

    // only the results of the sender are interesting
    if (this_pe == 0) {

        std::cout << std::setw(20) << "NBI_QUIET_ALL: " << time_nbi_quiet_all.count() << std::endl;
        std::cout << std::setw(20) << "NBI_QUIET_EACH: " << time_nbi_quiet_each.count() << std::endl;
        std::cout << std::setw(20) << "NBI_NO_QUIET: " << time_nbi_no_quiet.count() << std::endl;
        std::cout << std::setw(20) << "NBI_BLOCKING: " << time_blocking.count() << std::endl;
    }




//    for (uint64_t message_size = min_message_size; message_size <= max_message_size; message_size <<= 1) {
//        const auto time_taken = time_kernel(generalized_benchmark, grid_dim, block_dim, 0, stream,
//                                            data_source, data_sink, this_pe, count, message_size, data_separation);
//        if (this_pe == 0) {
//            uint64_t num_bytes = grid_dim * block_dim * message_size * count;
//            std::cout << "06_put_granularity"
//                      << "," << grid_dim
//                      << "," << block_dim
//                      << "," << num_hosts
//                      << "," << data_separation
//                      << "," << count // number of times that the message size is sent in total per kernel
//                      << "," << max_message_size
//                      << "," << message_size // size of a single send operation (with put nbi)
//                      << "," << num_bytes
//                      << "," << gb_per_sec(time_taken, num_bytes)
//                      << std::endl;
//        }
//    }

    nvshmem_free(data_source);
    nvshmem_free(data_sink);
    nvshmem_finalize();
    return 0;
}
