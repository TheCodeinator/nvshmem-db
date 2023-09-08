project(bench_08_tuple_scan LANGUAGES CXX CUDA)

add_executable(bench_08_tuple_scan tuple_scan.cu ../../src/shuffle_data_tmp.cu)
add_dependencies(bench_08_tuple_scan nvshmem-db)
target_link_libraries(bench_08_tuple_scan nvshmem-db)

set_property(TARGET bench_08_tuple_scan PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(bench_08_tuple_scan PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
