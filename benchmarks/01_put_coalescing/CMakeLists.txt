project(bench_01_put_coalescing LANGUAGES CXX CUDA)
add_executable(bench_01_put_coalescing nvshmem_put_coalescing.cu)
add_dependencies(bench_01_put_coalescing nvshmem-db)
target_link_libraries(bench_01_put_coalescing nvshmem-db)
set_property(TARGET bench_01_put_coalescing PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(bench_01_put_coalescing PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
