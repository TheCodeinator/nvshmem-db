add_executable(bench_02_shuffle bench_shuffle.cu)

target_link_libraries(bench_02_shuffle nvshmem)

set_property(TARGET bench_02_shuffle PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(bench_02_shuffle PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)


SET(TEST_COMMAND
        /opt/hydra/bin/nvshmrun -localhost c01.lab -f /home/amuth/hostfile -n 2
        $<TARGET_FILE:bench_02_shuffle>)

add_test(NAME bench_02_shuffle COMMAND ${TEST_COMMAND})
