# test executable
add_executable(helloworld TestNVShMemHelloWorld.cu)
target_include_directories(helloworld ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


add_library(nvshmem-db SomeLibraryInterfaceFunctions.cpp)

include_directories(${CMAKE_CURRENT_DIRECTORY} ${CMAKE_SOURCE_DIR})

target_include_directories(nvshmem-db PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(nvshmem-db PROPERTIES LINKER_LANGUAGE CUDA)
set_property(TARGET nvshmem-db PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(nvshmem-db PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)


# declare public interface of our library
set_target_properties(nvshmem-db PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/inc/SomeLibraryInterfaceFunctions.h)

if(DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -DDEBUG")
    target_compile_options(nvshmem-db PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-src-in-ptx;-lineinfo>")
    target_compile_definitions (nvshmem-db PRIVATE -DDEBUG)
endif()


