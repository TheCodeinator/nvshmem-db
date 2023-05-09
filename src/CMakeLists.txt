include_directories(${CMAKE_CURRENT_DIRECTORY})

add_library(nvshmem-db SomeTest.cpp)
target_include_directories(nvshmem-db PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(nvshmem-db PROPERTIES LINKER_LANGUAGE CUDA)
set_property(TARGET nvshmem-db PROPERTY POSITION_INDEPENDENT_CODE ON)

set_target_properties(nvshmem-db PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -DDEBUG")
    target_compile_options(nvshmem-db PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-src-in-ptx;-lineinfo>")
    target_compile_definitions (nvshmem-db PRIVATE -DDEBUG)
endif()


