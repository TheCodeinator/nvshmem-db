#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include <cuda.h>
#include <c++/10/array>
#include <c++/10/vector>
#include "nvshmem.h"
#include "Meas.cuh"
#include "Macros.cuh"
#include "NVSHMEMUtils.cuh"

// TODO: verify results make sense
// TODO: return results to CPU and print in csv format

constexpr size_t N_TESTS{3};

enum SendState {
    RUNNING = 0,
    FINISHED = 1,
};

__device__ void send_one_thread_sep(uint8_t *const data,
                                    const int other_pe,
                                    const uint64_t n_elems,
                                    const uint64_t n_iterations) {
    // send data to other PE at same position
    for (size_t it{0}; it < n_iterations; ++it) {
        for (size_t i{0}; i < n_elems; ++i) {
            nvshmem_uint8_put_nbi(data + i,
                                  data + i,
                                  1,
                                  other_pe);
        }
    }

    // make sure all send buffers are reusable
    nvshmem_quiet();
    // sync with receiver
    nvshmem_barrier_all();
}


__device__ void send_one_thread_once(uint8_t *const data,
                                     const int other_pe,
                                     const uint64_t n_elems,
                                     const uint64_t n_iterations) {
    // send data in one go
    for (size_t it{0}; it < n_iterations; ++it) {
        nvshmem_uint8_put_nbi(data,
                              data,
                              n_elems,
                              other_pe);
    }

    // make sure all send buffers are reusable
    nvshmem_quiet();
    // sync with receiver
    nvshmem_barrier_all();
}

__device__ void send_multi_thread_sep(uint8_t *const data,
                                      const int other_pe,
                                      const uint64_t n_elems,
                                      const uint64_t n_iterations) {
    const uint32_t thread_global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t thread_stride = blockDim.x * gridDim.y;

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

    // make sure all send buffers are reusable
    nvshmem_quiet();
    // sync with receiver
    if (thread_global_id == 0) {
        nvshmem_barrier_all();
    }
}

__device__ void recv(uint8_t *const data,
                     const int other_pe,
                     const uint64_t n_elems,
                     const uint64_t n_iterations) {
    Meas meas{};

    // wait for receiver to finish and sync with it
    nvshmem_barrier_all();
}

enum TestCase {
    one_thread_sep,
    one_thread_once,
    multi_thread_sep,
};

__global__ void exchange_data(int this_pe,
                              uint8_t *const data,
                              const uint64_t n_elems,
                              const uint64_t n_iterations,
                              TestCase test_case) {
    const int other_pe = static_cast<int>(!this_pe); // there are two PEs in total
    const uint32_t thread_global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t thread_stride = blockDim.x * gridDim.y;

    // PE 0 is the sender
    if (this_pe == 0) {
        switch (test_case) {
            case one_thread_sep:
                send_one_thread_sep(data, other_pe, n_elems, n_iterations);
                break;
            case one_thread_once:
                send_one_thread_once(data, other_pe, n_elems, n_iterations);
                break;
            case multi_thread_sep:
                send_multi_thread_sep(data, other_pe, n_elems, n_iterations);
                break;
        }
    } else { // PE 1 is the receiver
        // receiver does not do anything but waiting, only needs one thread in all scenarios
        if (thread_global_id == 0) {
                recv(data, other_pe, n_elems, n_iterations);
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

    // vector to store the measurements for the different kernel calls
    std::vector<std::chrono::microseconds> measurements{};
    measurements.reserve(N_TESTS);

    for (int t_case = one_thread_sep; t_case <= multi_thread_sep; ++t_case) {
        measurements.push_back(time_kernel(exchange_data, grid_dim, block_dim, 1024 * 4, stream,
                                           this_pe, data, n_elems, n_iterations, t_case));
    }

    // deallocate all the memory that has been allocated
    nvshmem_free(data);

    if (this_pe == 0) {
        for (const auto &meas : measurements) {
            std::cout << "," << gb_per_sec(meas, n_iterations * n_elems);
        }
        std::cout << std::endl;
    }
}
