#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include "nvshmem.h"
#include <vector>
#include <string>
#include "Macros.cuh"
#include "NVSHMEMUtils.cuh"

__global__ void send_benchmark(const int other_pe, uint32_t send_size, uint32_t total_size, uint8_t * sym_mem){

    int t = threadIdx.x;
    if(t==0){

        for(uint32_t i {0}; i<total_size; i+=send_size){
            nvshmem_uint8_put_nbi(sym_mem+i*send_size, i*send_size, send_size, other_pe );
        }
        nvshmem_quiet();
    }

}

int main(int argc, char* argv[]){

    // give size per send as argument
    assert(argc == 2);
    const size_t size_per_send = std::stoull(argv[1]);


}