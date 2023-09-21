#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include "nvshmem.h"
#include <vector>
#include <string>
#include "Macros.cuh"

/*
    Short running
*/
__global__ void calculate(size_t num_launches, int* res) {

    //c.f. calculate_long
    __nanosleep(1e9);
    *res += 1;
}

/*
    Long running kernel over whole domain
*/
__global__ void calculate_parts(size_t num_launches, int* res) {

    // Compute capability >= 7.0 (V100)
    __nanosleep(1e9/num_launches);
    *res += 1;

}

// args:
// 1: num_launches
int main(int argc, char *argv[]) {

    assert(argc == 2);
    const size_t num_launches = std::stoull(argv[1]);

    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t stream1;
    CUDA_CHECK(cudaStreamCreate(&stream1));

    int* res;
    CUDA_CHECK(cudaMalloc((void**)&res, sizeof(int)));
    CUDA_CHECK(cudaMemset(res, 0, sizeof(int)));

    // Warm up CUDA context
    calculate<<<1,1,0,stream1>>>(1,res);
    cudaStreamSynchronize(stream1);

    CUDA_CHECK(cudaMemset(res, 0, sizeof(int)));

    auto start = std::chrono::steady_clock::now();

    calculate<<<1,1,0,stream1>>>(1,res);
    cudaStreamSynchronize(stream1);

    auto stop = std::chrono::steady_clock::now();

    int* host_res;
    CUDA_CHECK(cudaMemcpy(host_res, res, sizeof(int), cudaMemcpyDeviceToHost));
    assert(*host_res == 1);

    auto dur = stop-start;
    auto t_ms = dur.count() * 1e-6;

    CUDA_CHECK(cudaMemset(res, 0, sizeof(int)));

    auto start2 = std::chrono::steady_clock::now();

    for(int i{0}; i<num_launches;i++) {
        calculate_parts<<<1, 1, 0, stream1>>>(num_launches, res);
        cudaStreamSynchronize(stream1);
    }

    auto stop2 = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaMemcpy(host_res, res, sizeof(int), cudaMemcpyDeviceToHost));
    assert(*host_res == num_launches);

    auto dur2 = stop2 - start2;
    auto t_ms2 = dur2.count() * 1e-6;

    std::cout << "type,launches,time_single,time_multi" << std::endl;
    std::cout << "05_single_multi_launch_simple" << "," << num_launches << "," << t_ms << "," << t_ms2 << std::endl;

    return EXIT_SUCCESS;
}