#ifndef NVSHMEM_DB_NVSHMEMUTILS_CUH
#define NVSHMEM_DB_NVSHMEMUTILS_CUH

#include <chrono>
#include "nvshmem.h"
#include "Macros.cuh"

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

inline __device__ uint32_t global_thread_id() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

inline __device__ uint32_t global_thread_count() {
    return gridDim.x * blockDim.x;
}

/**
 * wrapper for nvshmem collective launch API using variadic template parameters
 */
template<typename KernelFuncType, typename... Args>
void collective_launch(KernelFuncType kernel_func,
                       const dim3 grid_dim,
                       const dim3 block_dim,
                       const uint32_t shared_mem,
                       cudaStream_t cuda_stream,
                       const Args &... args) {
    // array of arguments to pass to nvshmem
    void *args_array[sizeof...(args)];

    // iterate over variadic template args to extract pointers of the variadic references and store them in the args_array
    {
        size_t i{0};
        ([&] {
            // first cast away const, as required by shitty nvshmem
            using arg_type = std::remove_reference_t<decltype(args)>;
            args_array[i++] = const_cast<
                    std::remove_const_t<arg_type> *
                    >(&args);
        }(), ...);
    }

    // launch using nvshmem's collective launch api
    NVSHMEM_CHECK(
            nvshmemx_collective_launch(reinterpret_cast<const void *>(kernel_func),
                                       grid_dim,
                                       block_dim,
                                       args_array,
                                       shared_mem,
                                       cuda_stream));
}


/**
 * times a given kernel
 * @param kernel_func the kernel function to time
 * @param grid_dim number of blocks
 * @param block_dim number of threads per block
 * @param shared_mem size of shared memeory passed to nvshmem_collective_launch
 * @param cuda_stream cuda stream passed to nvshmem_collective_launch
 * @param args const references arguments to pass to the kernel function
 */
template<typename KernelFuncType, typename grid_dim_t, typename block_dim_t, typename... Args>
std::chrono::nanoseconds time_kernel(KernelFuncType kernel_func,
                                     const grid_dim_t grid_dim,
                                     const block_dim_t block_dim,
                                     const uint32_t shared_mem,
                                     cudaStream_t cuda_stream,
                                     Args &&... args) {
    using namespace std::chrono;

    auto time_start = steady_clock::now();

    collective_launch(kernel_func, grid_dim, block_dim, shared_mem, cuda_stream, std::forward<Args>(args)...);

    // wait for kernel to finish
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

    return duration_cast<nanoseconds>(steady_clock::now() - time_start);
}

template<typename T>
__device__ void fucking_fcollect(nvshmem_team_t team, T *dest, const T *src, const size_t nelem)
{
    const auto pe_count = nvshmem_team_n_pes(team);
    const auto my_pe = nvshmem_team_my_pe(team);
    for(uint32_t pe = 0; pe < pe_count; ++pe) {
        //printf("PE %d, sending %llu elements of size %llu to %lu (dest: %p, src: %p)\n", nvshmem_team_my_pe(team), nelem, sizeof(T), pe, dest + (my_pe * nelem), src);
        nvshmem_putmem_nbi(dest + (my_pe * nelem), src, nelem * sizeof(T), pe);
    }
    nvshmem_quiet();
    nvshmem_barrier(team);
}

template<typename Rep, typename Period>
double gb_per_sec(std::chrono::duration<Rep, Period> time, const uint64_t bytes) {
    using namespace std::chrono;
    // the ratio of 1024^3 / 1000^3 which we have to use to convert bytes / nanosecond to GB/s
    constexpr double conversion_factor = 1.073741824;
    return (static_cast<double>(bytes) / duration_cast<nanoseconds>(time).count()) / conversion_factor;
};

consteval size_t log2const(size_t n) {
    return n == 1 ? 0 : 1 + log2const(n >> 1);
}

/**
 * use with case to avoid overflow and too much compile-time recursion.
 * Evaluated at compile time of possible
 */
constexpr size_t int_pow(size_t base, size_t power) {
    if (power == 1){
        return base;
    } else if (power == 0) {
        return 1;
    }

    return base * int_pow(base, power - 1);
}

#endif //NVSHMEM_DB_NVSHMEMUTILS_CUH
