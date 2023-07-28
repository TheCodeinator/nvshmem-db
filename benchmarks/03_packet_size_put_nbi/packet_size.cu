#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include <cuda.h>
#include <c++/10/array>
#include "nvshmem.h"
#include "Meas.cuh"
#include "Macros.cuh"

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

__device__ Meas time_send(uint8_t *const data,
                          const int other_pe,
                          uint32_t *const flag,
                          const uint64_t n_iterations,
                          const size_t msg_size,
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


    // send same data for specified number of iterations
    for (size_t it{0}; it < n_iterations; ++it) {
        nvshmem_uint8_put_nbi(
                data + thread_global_id, // use specific offset for each thread to not run into any data race conflicts
                data + thread_global_id,
                msg_size,
                other_pe);
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
                          const uint64_t n_iterations) {
    Meas meas{};

    // sync with other PE to make them start simultaneously
    nvshmem_barrier_all();

    meas.start = clock64();

    // wait until flag has been delivered, this then indicates all previous data has been delivered
    while (*flag == SendState::RUNNING);

    meas.stop = clock64();

    // sanity check: has received the data from the sender at least once
    assert(*data == 1);

    // reset receive buffer and flag for next test
    memset(data, 0, MAX_SEND_SIZE);
    *flag = SendState::RUNNING;

    return meas;
}

// list of message sizes to test
// send with nbi pu tin loop with configurable iterations
// nvshmem_quiet afterwards
// record time for each message size

__global__ void exchange_data(int this_pe,
                              uint8_t *const data,
                              uint32_t *const flag,
                              const uint64_t n_iterations,
                              Meas *const meas_dev) {
    const int other_pe = static_cast<int>(!this_pe); // there are two PEs in total
    const uint32_t thread_global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t thread_stride = blockDim.x * gridDim.y;

    // PE 0 is the sender
    if (this_pe == 0) {
        // populate data to send to PE 1
        for (size_t i{thread_global_id}; i < (MAX_SEND_SIZE * gridDim.x * blockDim.x); i += thread_stride) {
            // just write all ones
            data[i] = static_cast<uint8_t>(1);
        }
        __syncthreads();

        // set local flag to finished state on this PE, send flag every time the sender is finished
        // Receiver will reset its local instance of the flag after each of the tests
        *flag = SendState::FINISHED;

        // send msgs with exponentially increasing sizes starting from 2 and going to max infiniband packet size
        for (size_t test{0}; test < N_TESTS; ++test) {
            time_send(data, other_pe, flag, n_iterations, pow(2, test), meas_dev[test]);
        }

    } else { // PE 1 is the receiver
        // make reads from flag volatile
        volatile uint32_t *flag_vol = flag;

        // receiver does not do anything but waiting, only needs one thread in all scenarios
        if (thread_global_id == 0) {
            for (size_t i{0}; i < N_TESTS; ++i) {
                meas_dev[i] = time_recv(data, other_pe, flag_vol, n_iterations);
            }
        }
    }
}

// TODO: use host to measure time since GPU clock frq. can change dynamically and is therefore not reliable

/**
 * cmd arguments:
 * 0) program name (implicit)
 * 1) number of iterations
 * 2) grid dims
 * 3) block dims
 * 4) number of hosts
 */
int main(int argc, char *argv[]) {
    // init nvshmem
    int n_pes, this_pe;
    cudaStream_t stream;

    assert(argc == 5);
    const u_int64_t n_iterations = std::stoull(argv[1]);
    const u_int32_t grid_dim = stoi(argv[2]);
    const u_int32_t block_dim = stoi(argv[3]);
    const u_int32_t n_hosts = stoi(argv[4]);

    nvshmem_init();
    this_pe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    n_pes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    cudaSetDevice(this_pe);
    cudaStreamCreate(&stream);

    // this test is supposed to be executed on 2 PEs, each sends and receives data from the other PE
    assert(n_pes == 2);

    // allocate symmetric device memory for sending/receiving the data
    auto *const data = static_cast<uint8_t *>(nvshmem_malloc(MAX_SEND_SIZE * block_dim * grid_dim));
    auto *const flag = static_cast<int *>(nvshmem_malloc(sizeof(uint32_t)));

    // memory for storing the measurements and returning them from device to host
    Meas meas_host[N_TESTS];
    Meas *meas_dev;
    cudaMalloc(&meas_dev, sizeof(Meas) * N_TESTS);

    // call benchmarking kernel
    void *args[] = {&this_pe,
                    const_cast<uint8_t **>(&data),
                    const_cast<int **>(&flag),
                    const_cast<uint64_t *>(&n_iterations),
                    &meas_dev};
    NVSHMEM_CHECK(
            nvshmemx_collective_launch((const void *) exchange_data, grid_dim, block_dim, args, 1024 * 4, stream));

    // wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy results to host
    cudaMemcpy(meas_host, meas_dev, sizeof(Meas) * N_TESTS, cudaMemcpyDeviceToHost);

    // deallocate all the memory that has been allocated
    cudaFree(meas_dev);
    nvshmem_free(data);
    nvshmem_free(flag);

    if (this_pe == 0) {
        for (size_t i{0}; i < N_TESTS; ++i) {
            // send 2^i bytes in each iteration
            const auto n_bytes = pow(2, i);
            std::cout << "03_packet_size_put_nbi," << n_hosts
                      << "," << n_bytes
                      << "," << n_iterations
                      << "," << grid_dim
                      << "," << block_dim
                      << "," << meas_host[i].get_throughput(n_iterations * n_bytes) * grid_dim * block_dim << std::endl;
        }
    }
}
