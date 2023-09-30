#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <assert.h>
#include "Macros.cuh"

 __constant__ uint32_t work_size = 1000;


enum class OccupancyMode {
    SLEEP = 0,
    LOOP = 1
};

/*
    Short running
*/
template<OccupancyMode occupancy>
__global__ void calculate(size_t num_launches, int* res, double* approx) {

    if constexpr (occupancy == OccupancyMode::SLEEP) {
        //c.f. calculate_long
        __nanosleep(1000000000U);
        *res += 1;
    }
    else if constexpr (occupancy == OccupancyMode::LOOP){
        // Approximate pi/4 https://en.wikipedia.org/wiki/Leibniz_formula_for_π
        for(uint32_t i {0}; i<work_size*num_launches; i++){
            *approx += pow((-1),i)/(2*i+1);
        }
        *res += 1;
    }

}

/*
    Long running kernel over whole domain
*/
template<OccupancyMode occupancy>
__global__ void calculate_parts(size_t num_launches, int* res, double* approx) {

    if constexpr (occupancy == OccupancyMode::SLEEP) {
        // Compute capability >= 7.0 (V100)
        __nanosleep(100 / num_launches);
        *res += 1;
    }
    else if constexpr (occupancy == OccupancyMode::LOOP){
        // Approximate pi/4 https://en.wikipedia.org/wiki/Leibniz_formula_for_π
        for(uint32_t i {0}; i<work_size; i++){
            *approx += pow((-1),i)/(2*i+1);
        }
        *res += 1;
    }
}

// args:
// 1: num_launches
int main(int argc, char *argv[]) {

    assert(argc == 2);
    const uint32_t num_launches = std::stoull(argv[1]);

    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t stream1;
    CUDA_CHECK(cudaStreamCreate(&stream1));

    int* res;
    double* approx;
    CUDA_CHECK(cudaMalloc((void**)&res, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&approx, sizeof(double)));
    CUDA_CHECK(cudaMemset(res, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(approx, 0.0, sizeof(double)));

    // Warm up CUDA context
    calculate<OccupancyMode::SLEEP><<<1,1,0,stream1>>>(num_launches,res, approx);
    cudaStreamSynchronize(stream1);

    CUDA_CHECK(cudaMemset(res, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(approx, 0.0, sizeof(double)));

    auto start = std::chrono::steady_clock::now();

    calculate<OccupancyMode::LOOP><<<1,1,0,stream1>>>(num_launches,res, approx);
    cudaStreamSynchronize(stream1);

    auto stop = std::chrono::steady_clock::now();

    int* host_res = reinterpret_cast<int*>(malloc(sizeof(int)));
    double* host_approx = reinterpret_cast<double*>(malloc(sizeof(double)));
    CUDA_CHECK(cudaMemcpy(host_res, res, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_approx, approx, sizeof(double), cudaMemcpyDeviceToHost));
    assert(*host_res == 1);

    auto dur = stop-start;
    auto t_ms = dur.count() * 1e-6;

    CUDA_CHECK(cudaMemset(res, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(approx, 0.0, sizeof(double)));

    auto start2 = std::chrono::steady_clock::now();

    for(int i{0}; i<num_launches;i++) {
        calculate_parts<OccupancyMode::LOOP><<<1, 1, 0, stream1>>>(num_launches, res, approx);
        cudaStreamSynchronize(stream1);
    }

    auto stop2 = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaMemcpy(host_res, res, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_approx, approx, sizeof(double), cudaMemcpyDeviceToHost));
    assert(*host_res == num_launches);

    auto dur2 = stop2 - start2;
    auto t_ms2 = dur2.count() * 1e-6;

    std::cout << "05_single_multi_launch_simple" << "," << num_launches << "," << t_ms << "," << t_ms2 << std::endl;

    return EXIT_SUCCESS;
}
