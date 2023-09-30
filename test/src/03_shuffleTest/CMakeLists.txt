add_executable(03_shuffleTest shuffle_test.cu)

target_link_libraries(03_shuffleTest nvshmem)

set_property(TARGET 03_shuffleTest PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(03_shuffleTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

SET(TEST_COMMAND
        /opt/hydra/bin/nvshmrun -localhost c01.lab -f /home/amuth/hostfile -n 2
        $<TARGET_FILE:03_shuffleTest>)

#SET(TEST_COMMAND
#        /opt/hydra/bin/nvshmrun -localhost 172.18.94.10 172.18.94.11 -n 2
#        $<TARGET_FILE:03_shuffleTest>)

add_test(NAME 03_shuffleTest COMMAND ${TEST_COMMAND})
