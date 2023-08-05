#ifndef NVSHMEM_DB_NVSHMEMUTILS_CUH
#define NVSHMEM_DB_NVSHMEMUTILS_CUH

#include <chrono>
#include "Macros.cuh"

/**
 * wrapper for nvshmem collective launch API using variadic template parameters
 */
template<typename KernelFuncType, typename... Args>
void collective_launch(KernelFuncType kernel_func,
                       const uint32_t grid_dim,
                       const uint32_t block_dim,
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
template<typename KernelFuncType, typename... Args>
std::chrono::microseconds time_kernel(KernelFuncType kernel_func,
                                      const uint32_t grid_dim,
                                      const uint32_t block_dim,
                                      const uint32_t shared_mem,
                                      cudaStream_t cuda_stream,
                                      Args &&... args) {
    using namespace std::chrono;

    auto time_start = system_clock::now();

    collective_launch(kernel_func, grid_dim, block_dim, shared_mem, cuda_stream, std::forward<Args>(args)...);

    // wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    return duration_cast<microseconds>(system_clock::now() - time_start);
}

template<typename Rep, typename Period>
double gb_per_sec(std::chrono::duration<Rep, Period> time, const uint64_t bytes) {
    using namespace std::chrono;
    return (static_cast<double>(bytes) / 1000) / duration_cast<microseconds>(time).count();
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
