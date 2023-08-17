#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include <cuda.h>
#include <vector>
#include <variant>
#include "nvshmem.h"
#include "NVSHMEMUtils.cuh"
#include "Macros.cuh"

constexpr long long MAX_SEND_SIZE{4096};

// TODO: verify results make sense and benchmark code is bug-free

// from 2 go up to the max packet size in exponential steps
constexpr size_t N_TESTS{log2const(MAX_SEND_SIZE) + 1};

__device__ void send(uint8_t *const data,
                     const int other_pe,
                     const uint64_t n_elems,
                     const size_t msg_size) {
    const uint32_t thread_global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // start for-loop together
    __syncthreads();

    // when having more threads or bigger messages, we need less loop iterations because we send more in one iteration
    // we send the same data in every iteration to avoid memory size limitations
    for (size_t it{0}; it < n_elems / (blockDim.x * gridDim.x * msg_size); ++it) {
        nvshmem_uint8_put_nbi(
                data + MAX_SEND_SIZE * thread_global_id, // use specific offset for each thread to not run into any data race conflicts
                data + MAX_SEND_SIZE * thread_global_id,
                msg_size,
                other_pe);
    }

    // make sure all send buffers are reusable and sync with receiver
    nvshmem_quiet();
    if (thread_global_id == 0) {
        nvshmem_barrier_all();
    }
}

__device__ void recv(uint8_t *const data) {
    // wait for sender to be finished
    nvshmem_barrier_all();
}

__global__ void exchange_data(int this_pe,
                              uint8_t *const data,
                              const uint64_t n_elems,
                              const uint64_t msg_size) {
    const int other_pe = static_cast<int>(!this_pe); // there are two PEs in total
    const uint32_t thread_global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // PE 0 is the sender
    if (this_pe == 0) {
        send(data, other_pe, n_elems, msg_size);

    } else { // PE 1 is the receiver
        // receiver does not do anything but waiting, only needs one thread in all scenarios
        if (thread_global_id == 0) {
            recv(data);
        }
    }
}

// Do a barrier operation to prevent compile from optimizing out empty kernel
__global__ void warmup(){

	const uint32_t thread_global_id = blockIdx.x * blockDim.x + threadIdx.x;

	if(thread_global_id == 0) {
		nvshmem_barrier_all();
	}

}


/**
 * cmd arguments:
 * 0) program name (implicit)
 * 1) number of bytes
 * 2) grid dims
 * 3) block dims
 * 4) number of hosts
 */
int main(int argc, char *argv[]) {
    // init nvshmem
    int n_pes, this_pe;
    cudaStream_t stream;

    assert(argc == 5);
    const u_int64_t n_elems = std::stoull(argv[1]);
    const u_int32_t grid_dim = stoi(argv[2]);
    const u_int32_t block_dim = stoi(argv[3]);
    const u_int32_t n_hosts = stoi(argv[4]);

    nvshmem_init();
    this_pe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    n_pes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    cudaSetDevice(this_pe);
    cudaStreamCreate(&stream);

    if (n_pes != 2) {
        throw std::logic_error("This test has to be started with exactly 2 PEs.");
    }

    if(n_elems / (block_dim * grid_dim * MAX_SEND_SIZE) < 1 || n_elems % (block_dim * grid_dim * MAX_SEND_SIZE) != 0) {
        throw std::logic_error(
                "Make sure that the number of elements is a multiple of the product of the total number of threads and the maximum msg size (" +
                std::to_string(MAX_SEND_SIZE) + ").");
    }

    // allocate symmetric device memory for sending/receiving the data
    auto *const data = static_cast<uint8_t *>(nvshmem_malloc(MAX_SEND_SIZE * block_dim * grid_dim));

    std::vector<std::pair<uint32_t, std::chrono::nanoseconds>> measurements{};
    measurements.reserve(N_TESTS);

    // do not time warmup
    collective_launch(warmup, 1, 1, 1024, stream);
   
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // send msgs with exponentially increasing sizes starting from 2
    for (size_t test{0}; test < N_TESTS; ++test) {
        const auto msg_size = static_cast<uint64_t>(int_pow(2, test));
        measurements.emplace_back(msg_size,
                                  time_kernel(exchange_data, grid_dim, block_dim, 1024 * 4, stream,
                                              this_pe, data, n_elems, msg_size));
    }

    // deallocate all the memory that has been allocated
    nvshmem_free(data);

    if (this_pe == 0) {
        for (const auto &meas: measurements) {
            // send 2^i bytes in each iteration
            std::cout << "03_packet_size_put_nbi," << n_hosts
                      << "," << meas.first // message size
                      << "," << n_elems
                      << "," << grid_dim
                      << "," << block_dim
                      << "," << gb_per_sec(meas.second /* nanoseconds */, n_elems) << std::endl;
        }
    }
}
