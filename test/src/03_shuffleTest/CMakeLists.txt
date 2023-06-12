
add_executable(03_shuffleTest shuffleTest.cu)
add_dependencies(03_shuffleTest nvshmem-db)

target_link_libraries(03_shuffleTest nvshmem-db)

set_property(TARGET 03_shuffleTest PROPERTY POSITION_INDEPENDENT_CODE ON)
#set_target_properties(03_shuffleTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_test(NAME 03_shuffleTest COMMAND 03_shuffleTest)

if (DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -DDEBUG")
    target_compile_options(03_shuffleTest PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-src-in-ptx;-lineinfo>")
    target_compile_definitions(03_shuffleTest PRIVATE -DDEBUG)
endif ()
