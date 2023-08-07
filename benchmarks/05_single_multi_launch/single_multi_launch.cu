#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include "nvshmem.h"
#include <vector>
#include <string>
#include "rdma.hpp"
#include "Macros.cuh"

/*
    Emulate calculation of sending destination and writing in output buffer
*/
__global__ void calculate(const uint32_t * in, uint32_t * buff, uint32_t size_in, size_t size_in, size_t size_buff){

    //c.f. calculate_and_send
    __nanosleep(1e9);
    memcpy(buff, in, size_buff);

}

/*
    Emulate calculation of sending destination and sending to remotes
*/
__global__ void calculate_and_send(const uint32_t* in, uint32_t* buff, uint32_t size_in,
                                   uint32_t size_buf, uint32_t * sym_mem, uint32_t this_pe){

    for(uint32_t off{0}; off<size_in; off+=size_buf){

        // Compute capability >= 7.0 (V100)
        __nanosleep(1e9);

        nvshmem_uint32_put_nbi(sym_mem, in+off, size_buf, this_pe%1);

        nvshmem_quiet();

    }
}


int main(int argc, char* argv[]){

    // create cuda streams for alternating in new kernel launches
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // get nvshmem environment information
    uint32_t this_pe, n_pes;
    this_pe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    n_pes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);

    // rdma environment information
    std::vector<string> ips;
    string ip = "";
    uint32_t rdma_port = 0;

    size_t size_in;
    auto* in = static_cast<uint32_t *>(cudaMalloc(size_in*sizeof(uint32_t)));
    cudaMemset(in, 1, size_in*sizeof(uint32_t));

    size_t size_buff;
    auto* buff = static_cast<uint32_t *>(cudaMalloc(size_buff*sizeof(uint32_t)));
    auto* sym_mem = static_cast<uint32_t *>(nvshmem_malloc(buff_size*sizeof(uint32_t)));

    // Make RDMA connection

    rdma::Server server{ip, rdma_port};
    rdma::Client client{ip, rdma_port};

    std::vector<rdma::Connection*> conns;

    for(uint32_t i{0}, i<ips.size(), i++){
        conns.push_back(client.connect_to(i, rdma_port));
    }

    // Warm up CUDA context
    calculate<<<1,1>>>(in, buff, size_in, size_buff);

    std::ofstream outfile;
    outfile.open("results.csv");
    outfile << "type, node_count, in n, send n, out n" << std::endl;

    void* args[] = {in, buff, size_in, size_buff, sym_mem, this_pe};
    auto start = std::chrono::steady_clock::now();
    nvshmemx_collective_launch(&calculate_and_send, 1, 1, args, size_buff, stream1);
    cudaStreamSynchronize(stream1);
    auto stop = std::chrono::steady_clock::now();
    auto dur = stop-start;
    auto t_ms = dur.count()*1e-6;
    auto start2 = std::chrono::steady_clock::now();
    for(auto i{0}; i<size_in; i+=2*size_buff){
        calculate<<<1,1,0,stream1>>>();
        cudaStreamSynchronize(stream1);
        calculate<<<1,1,0,stream2>>>();

        //rdma 1

        cudaStreamSynchronize(stream2);

        //rdma 2

    }
    auto stop2 = std::chrono::steady_clock::now();
    auto dur2 = stop2-start2;
    auto t_ms2 = dur.count()*1e-6;

    outfile.close();

    return EXIT_SUCCESS;

}