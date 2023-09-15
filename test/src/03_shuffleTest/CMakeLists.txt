add_executable(03_shuffleTest shuffle_test.cu)
add_dependencies(03_shuffleTest nvshmem-db)

target_link_libraries(03_shuffleTest nvshmem-db)

set_property(TARGET 03_shuffleTest PROPERTY POSITION_INDEPENDENT_CODE ON)

SET(TEST_COMMAND
        /opt/hydra/bin/nvshmrun -localhost c01.lab -f /home/amuth/hostfile -n 4
        $<TARGET_FILE:03_shuffleTest>)

add_test(NAME 03_shuffleTest COMMAND ${TEST_COMMAND})

