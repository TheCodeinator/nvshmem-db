#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include "nvshmem.h"

// used to check the status code of cuda routines for errors
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

/*
    Emulate calculation of sending destination and writing in output buffer
*/
__global__ void calculate(const uint32_t * in, uint32_t * buff, int size){

}

/*
    Emulate calculation of sending destination and sending to remotes
*/
__global__ void calculate_and_send(const uint32_t* in, uint32_t* buff, int size){

}

int main(int argc, char* argv[]){

    std::ofstream outfile;
    outfile.open("results.csv");
    outfile << "type, node_count,in n,out n" << std::endl;



    outfile.close();

    return EXIT_SUCCESS;
}