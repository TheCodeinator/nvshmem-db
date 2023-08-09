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

/*
    Emulate calculation of sending destination and writing in output buffer
*/
__global__ void calculate(const uint32_t * in, uint32_t * buff, size_t size_buff){

    //c.f. calculate_and_send
    __nanosleep(1e9);
    memcpy(buff, in, size_buff*sizeof(uint32_t));

}

/*
    Emulate calculation of sending destination and sending to remotes
*/
__global__ void calculate_and_send(const uint32_t* in, uint32_t* buff, uint32_t size_in,
                                   uint32_t size_buf, uint32_t * sym_mem, uint32_t this_pe){

    for(uint32_t off{0}; off<size_in; off+=size_buf){

        // Compute capability >= 7.0 (V100)
        __nanosleep(1e9);

        nvshmem_uint32_put_nbi(sym_mem, in+off, size_buf, 1-this_pe);

    }
    nvshmem_quiet();
}


int main(int argc, char* argv[]){

    assert(argc == 4);
    const size_t size_in = std::stoull(argv[1]);
    const std::string ip1 = argv[2];
    const std::string ip2 = argv[3];

    // create cuda streams for alternating in new kernel launches
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // get nvshmem environment information
    nvshmem_init();
    uint32_t this_pe, other_pe, n_pes;

    this_pe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);

    n_pes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);

    other_pe = 1 - this_pe;

    if (n_pes != 2) {
        throw std::logic_error("This test has to be started with exactly 2 PEs.");
    }

    // rdma environment information
    std::vector<std::string> ips {ip1, ip2};
    string my_ip = ips[this_pe];

    constexpr uint32_t rdma_port = 5000;

    uint32_t * in = {};
    cudaMalloc((void**)in, 2*size_in*sizeof(uint32_t));
    cudaMemset(in, 1, size_in*sizeof(uint32_t));

    size_t size_buff = 4096;
    uint32_t * buff = {};
    cudaMalloc((void**)buff, size_buff*sizeof(uint32_t));
    uint32_t * sym_mem = reinterpret_cast<uint32_t *>(nvshmem_malloc(size_buff*sizeof(uint32_t)));

    // Make RDMA connection

    rdma::RDMA server {my_ip, rdma_port};

    // register RDMA memory

    server.register_memory(in, size_buff*sizeof(uint32_t));
    server.listen(rdma::RDMA::CLOSE_AFTER_LAST | rdma::RDMA::IN_BACKGROUND);

    rdma::Connection* conn = server.connect_to(ips[other_pe],rdma_port);

    // Warm up CUDA context

    calculate<<<1,1>>>(in, buff, size_buff);

    auto start = std::chrono::steady_clock::now();

    time_kernel(calculate_and_send, 1, 1, size_buff, stream1, in, buff, size_in, size_buff, sym_mem, this_pe);
    cudaStreamSynchronize(stream1);

    auto stop = std::chrono::steady_clock::now();

    auto dur = stop-start;
    auto t_ms = dur.count()*1e-6;

    auto start2 = std::chrono::steady_clock::now();


    for(auto i{0}; i<size_in; i+=2*size_buff){
        calculate<<<1,1,0,stream1>>>(in+i, buff, size_buff);
        cudaStreamSynchronize(stream1);
        calculate<<<1,1,0,stream2>>>(in+i+size_buff, buff, size_buff);
        conn->write(in+i,
                    size_buff*sizeof(uint32_t),
                    size_in*sizeof(uint32_t)+i*sizeof(uint32_t),
                    rdma::Flags().signaled());
        cudaStreamSynchronize(stream2);
        conn->write(in+i+size_buff,
                    size_buff*sizeof(uint32_t),
                    size_in*sizeof(uint32_t)+i*sizeof(uint32_t)+size_buff*sizeof(uint32_t),
                    rdma::Flags().signaled());
    }

    // wait till all writes are finished
    conn->sync_signaled();
    auto stop2 = std::chrono::steady_clock::now();

    auto dur2 = stop2-start2;
    auto t_ms2 = dur.count()*1e-6;

    auto num_launches = size_in/size_buff;

    std::cout << "05_single_multi_launch" << "," << size_in << "," << num_launches << "," << t_ms << "," << t_ms2 << std::endl;

    nvshmem_free(sym_mem);
    nvshmem_finalize();

    return EXIT_SUCCESS;

}