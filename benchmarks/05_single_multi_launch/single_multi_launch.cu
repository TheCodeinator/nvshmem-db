#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include "nvshmem.h"
#include <vector>
#include <string>
#include "rdma.hpp"
#include "Macros.cuh"
#include "NVSHMEMUtils.cuh"
#include "my_asserts.hpp"

/*
    Emulate calculation of sending destination and writing in output buffer
*/
__global__ void calculate(const uint32_t *in, uint32_t *buff, size_t size_buff) {

    //c.f. calculate_and_send
    __nanosleep(1e9);
    for (uint32_t i{0}; i < size_buff; i++) {
        buff[i] = in[i];
    }

}

/*
    Emulate calculation of sending destination and sending to remotes
*/
__global__ void calculate_and_send(const uint32_t *in, uint32_t size_in,
                                   uint32_t size_buff, uint32_t *sym_mem, uint32_t this_pe) {

    for (uint32_t off{0}; off < size_in; off += size_buff) {

        // Compute capability >= 7.0 (V100)
        __nanosleep(1e9);
        for (uint32_t i{0}; i < size_buff; i++) {
            sym_mem[off + i] = in[off + i];
        }
        nvshmem_uint32_put_nbi(sym_mem + off, sym_mem + size_in + off, size_buff, 1 - this_pe);

    }
    // Exit upon completion of all put calls
    nvshmem_quiet();
}

// args:
// size in bytes to send
// ip of NIC 1
// ip of NIC 2
int main(int argc, char *argv[]) {

    assert(argc == 4);
    const size_t size_in = std::stoull(argv[1]);
    const std::string ip1 = argv[2];
    const std::string ip2 = argv[3];

    // get nvshmem environment information
    nvshmem_init();
    uint32_t this_pe, other_pe, n_pes;

    this_pe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    n_pes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    other_pe = 1 - this_pe;

    if (n_pes != 2) {
        throw std::logic_error("This test has to be started with exactly 2 PEs.");
    }

    CUDA_CHECK(cudaSetDevice(this_pe));

    // create cuda streams for alternating in new kernel launches
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // rdma environment information
    std::string my_ip = argv[2 + this_pe];
    std::string other_ip = argv[2 + other_pe];

    constexpr uint32_t rdma_port = 5432;

    uint32_t *in;
    CUDA_CHECK(cudaMalloc((void **) &in, size_in * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(in, 1, size_in * sizeof(uint32_t)));

    size_t size_buff = 4096;

    // for use with multiple memory regions
    //uint32_t * buff1;
    //uint32_t * buff2;
    //CUDA_CHECK(cudaMalloc((void**)&buff1, 2*size_buff*sizeof(uint32_t)));
    //CUDA_CHECK(cudaMalloc((void**)&buff2, 2*size_buff*sizeof(uint32_t)));

    uint32_t *buff;
    CUDA_CHECK(cudaMalloc((void **) &buff, 4 * size_buff * sizeof(uint32_t)));

    uint32_t *sym_mem = reinterpret_cast<uint32_t *>(nvshmem_malloc(2 * size_in * sizeof(uint32_t)));

    // Warm up CUDA context
    calculate<<<1, 1>>>(in, buff, size_buff);

    auto dur = time_kernel(calculate_and_send, 1, 1, size_buff, stream1,
                           in, size_in, size_buff, sym_mem, this_pe);

    auto t_ms = dur.count() * 1e-6;

    // Make RDMA connection
    rdma::RDMA server{my_ip, rdma_port};

    // register RDMA memory

    // not yet supported
    // memory region1
    //server.register_memory(buff1, 2*size_buff*sizeof(uint32_t));
    // memory region2
    //server.register_memory(buff2, 2*size_buff*sizeof(uint32_t));

    server.register_memory(buff, 4 * size_buff * sizeof(uint32_t));

    server.listen(rdma::RDMA::CLOSE_AFTER_LAST | rdma::RDMA::IN_BACKGROUND);

    std::cout << "Listening on " << my_ip << ":" << rdma_port << " with NIC on socket " << server.numa_socket
              << std::endl;

    // wait for discovery
    sleep(10);

    std::cout << my_ip << " on socket " << server.numa_socket << " trying to connect to " << other_ip << std::endl;

        std::chrono::steady_clock::time_point start2;
        std::chrono::steady_clock::time_point stop2;

    {
        rdma::RDMA client{my_ip, rdma_port};
        const int size = 4 * size_buff * sizeof(uint32_t);
//        void *mem = malloc(size);
        void* mem;
        CUDA_CHECK(cudaMalloc(&mem, 4 * size_buff * sizeof(uint32_t)));
        client.register_memory(mem, size); // use std::span?

        rdma::Connection *conn = client.connect_to(other_ip, rdma_port);
        std::cout << "Start " << my_ip << std::endl;

        start2 = std::chrono::steady_clock::now();

        // mr_id only important with buff in different memory regions
        for (auto i{0}; i < size_in; i += 2 * size_buff) {
            std::cout << my_ip << " " << i << std::endl;
            calculate<<<1, 1, 0, stream1>>>(in + i, buff, size_buff);
            cudaStreamSynchronize(stream1);
            conn->write(buff,
                        size_buff * sizeof(uint32_t),
                        2 * size_buff * sizeof(uint32_t),
                        rdma::Flags().signaled(),
                        0);
            if (i != 0) { conn->sync_signaled(1); }
            calculate<<<1, 1, 0, stream2>>>(in + i + size_buff, buff + size_buff, size_buff);
            cudaStreamSynchronize(stream2);
            conn->write(buff + size_buff,
                        size_buff * sizeof(uint32_t),
                        3 * size_buff * sizeof(uint32_t),
                        rdma::Flags().signaled(),
                        0);
            conn->sync_signaled(1);
        }

        // wait till all writes are finished
        conn->sync_signaled();
        stop2 = std::chrono::steady_clock::now();

        client.close(conn);

    }

    server.wait();

    std::cout << "Stop " << my_ip << std::endl;

    auto dur2 = stop2 - start2;
    auto t_ms2 = dur2.count() * 1e-6;

    auto num_launches = size_in / size_buff;

    std::cout << "05_single_multi_launch" << "," << size_in << "," << num_launches << "," << t_ms << "," << t_ms2
              << std::endl;

    nvshmem_free(sym_mem);
    nvshmem_finalize();

    return EXIT_SUCCESS;
}
