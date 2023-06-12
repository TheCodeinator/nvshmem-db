
add_executable(02_nvshmemHelloWorld nvshmemHelloWorld.cu)

target_link_libraries(02_nvshmemHelloWorld nvshmem)

set_property(TARGET 02_nvshmemHelloWorld PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(02_nvshmemHelloWorld PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_test(NAME 02_nvshmemHelloWorld COMMAND 02_nvshmemHelloWorld)

if (DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -DDEBUG")
    target_compile_options(02_nvshmemHelloWorld PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-src-in-ptx;-lineinfo>")
    target_compile_definitions(02_nvshmemHelloWorld PRIVATE -DDEBUG)
endif ()