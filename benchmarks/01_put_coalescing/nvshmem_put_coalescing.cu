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

constexpr size_t N_ELEMS{1024 * 1024};
constexpr int TEST_1_SEND_DONE{1};
constexpr int TEST_2_SEND_DONE{2};

__global__ void exchange_data(int this_pe,
                              uint8_t *const data,
                              int *const flag) {
    const int other_pe = static_cast<int>(!this_pe); // there are two PEs in total

    // we only use one thread
    if (threadIdx.x != 0) {
        return;
    }

    // PE 0 is the sender
    if (this_pe == 0) {
        // populate data to send to PE 1
        for (size_t i{0}; i < N_ELEMS; ++i) {
            // write lower bits of index to every element
            data[i] = static_cast<uint8_t >(i);
        }

        // sync with other PE to make them start simultaneously
        nvshmem_barrier_all();

        long long start_time = clock64();

        // send data to other PE at same position
        for (size_t i{0}; i < N_ELEMS; ++i) {
            nvshmem_uint8_put_nbi(data + i,
                                  data + i,
                                  1,
                                  other_pe);
        }

        // let following mem operations be executed after the previous sending
        nvshmem_fence();

        // atomically set memory flag at other PE to signal that all previous send operation must have been completed (see fence)
        nvshmem_int_atomic_set(flag, TEST_1_SEND_DONE, 1);

        // make sure all send buffers are reusable
        nvshmem_quiet();

        long long stop_time = clock64();
        long long elapsed_time = stop_time - start_time;

        printf("Sender: time for sending %lu elems separately and calling nvshmem_quiet: %lld (clock start %lld, clock stop %lld)\n", N_ELEMS, elapsed_time, start_time, stop_time);

        // TODO: return result in CSV format

        // synchronize with other PE to make them start next test simultaneously
        nvshmem_barrier_all();

        start_time = clock64();

        // send data in one go
        nvshmem_uint8_put_nbi(data,
                              data,
                              N_ELEMS,
                              other_pe);

        // let following mem operations be executed after the previous sending
        nvshmem_fence();

        // atomically set memory flag at other PE to signal that all previous send operation must have been completed (see fence)
        nvshmem_int_atomic_set(flag, TEST_2_SEND_DONE, 1);

        // make sure all send buffers are reusable
        nvshmem_quiet();

        stop_time = clock64();
        elapsed_time = stop_time - start_time;

        printf("Sender: time for sending %lu elems at once and calling nvshmem_quiet: %lld (clock start %lld, clock stop %lld)\n", N_ELEMS, elapsed_time, start_time, stop_time);

    } else { // PE 1 is the receiver
        // sync with other PE to make them start simultaneously
        nvshmem_barrier_all();

        long long start_time = clock64();

        // wait until flag has been delivered, this then indicates all previous data has been delivered
        nvshmemi_wait_until(flag, NVSHMEM_CMP_EQ, TEST_1_SEND_DONE);

        long long stop_time = clock64();
        auto elapsed_time = stop_time - start_time;

        printf("Receiver: time until all %lu elems have been received separately: %lld (clock start %lld, clock stop %lld)\n", N_ELEMS, elapsed_time, start_time, stop_time);

        // verify correctness
        for (size_t i{0}; i < N_ELEMS; ++i) {
            // write lower bits of index to every element
            assert(data[i] == static_cast<uint8_t>(i));
        }

        // TODO: emit time in proper CSV format

        // reset receive buffer for next test
        memset(data, 0, N_ELEMS);

        // synchronize with other PE to make them start next test simultaneously
        nvshmem_barrier_all();

        start_time = clock64();

        // wait until flag has been delivered, this then indicates all previous data has been delivered
        nvshmemi_wait_until(flag, NVSHMEM_CMP_EQ, TEST_2_SEND_DONE);

        stop_time = clock64();
        elapsed_time = stop_time - start_time;

        printf("Receiver: time until all %lu elems have been received at once: %lld (clock start %lld, clock stop %lld)\n", N_ELEMS, elapsed_time, start_time, stop_time);

        // verify correctness
        for (size_t i{0}; i < N_ELEMS; ++i) {
            // write lower bits of index to every element
            assert(data[i] == static_cast<uint8_t>(i));
        }

        // TODO: emit time in proper CSV format

    }
}

int main(int argc, char *argv[]) {
    // init nvshmem
    int n_pes, this_pe;
    cudaStream_t stream;

    nvshmem_init();
    this_pe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    n_pes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    printf("Hello from PE %d of %d\n", this_pe, n_pes);
    cudaSetDevice(this_pe);
    cudaStreamCreate(&stream);

    // this test is supposed to be executed on 2 PEs, each sends and receives data from the other PE
    assert(n_pes == 2);

    // allocate symmetric device memory for sending/receiving the data
    auto *const data = static_cast<uint8_t *>(nvshmem_malloc(N_ELEMS));
    auto *const flag = static_cast<int *>(nvshmem_malloc(sizeof(int)));

    // call benchmarking kernel
    void *args[] = {&this_pe,
                    const_cast<uint8_t **>(&data),
                    const_cast<int **>(&flag)};
    NVSHMEM_CHECK(nvshmemx_collective_launch((const void *) exchange_data, 1, 1, args, 1024 * 4, stream));

    // wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());


    // TODO: print results in suitable CSV format

//    std::ofstream outfile;
//    outfile.open("results.csv");
//    outfile << "type, node_count,in n,out n" << std::endl;
//
//    outfile.close();
}
