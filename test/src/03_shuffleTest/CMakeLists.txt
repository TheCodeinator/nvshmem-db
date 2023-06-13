
add_executable(03_shuffleTest shuffleTest.cu)
add_dependencies(03_shuffleTest nvshmem-db)

target_link_libraries(03_shuffleTest nvshmem-db)

set_property(TARGET 03_shuffleTest PROPERTY POSITION_INDEPENDENT_CODE ON)

add_test(NAME 03_shuffleTest COMMAND 03_shuffleTest)