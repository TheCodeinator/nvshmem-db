project(bench_03_packet_size_put_nbi LANGUAGES CXX CUDA)
add_executable(bench_03_packet_size_put_nbi packet_size.cu)
add_dependencies(bench_03_packet_size_put_nbi nvshmem-db)
target_link_libraries(bench_03_packet_size_put_nbi nvshmem-db)
set_property(TARGET bench_03_packet_size_put_nbi PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(bench_03_packet_size_put_nbi PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
