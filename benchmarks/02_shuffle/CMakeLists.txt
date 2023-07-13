add_executable(bench_02_shuffle bench_shuffle.cu)
add_dependencies(bench_02_shuffle nvshmem-db)
target_link_libraries(bench_02_shuffle nvshmem-db)
set_property(TARGET bench_02_shuffle PROPERTY POSITION_INDEPENDENT_CODE ON)
SET(TEST_COMMAND
        nvshmrun -np 2
        $<TARGET_FILE:bench_02_shuffle>)

add_test(NAME bench_02_shuffle COMMAND ${TEST_COMMAND})
