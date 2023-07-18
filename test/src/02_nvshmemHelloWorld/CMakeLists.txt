
add_executable(02_nvshmemHelloWorld nvshmem_hello_world.cu)

target_link_libraries(02_nvshmemHelloWorld nvshmem)

set_property(TARGET 02_nvshmemHelloWorld PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(02_nvshmemHelloWorld PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

SET(TEST_COMMAND
        /opt/hydra/bin/nvshmrun -localhost c01.lab -f /home/amuth/hostfile -n 2
        $<TARGET_FILE:02_nvshmemHelloWorld>)

add_test(NAME 02_nvshmemHelloWorld COMMAND ${TEST_COMMAND})
