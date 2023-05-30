# test executable

add_executable(helloworld TestNVShMemHelloWorld.cu)
target_include_directories(helloworld PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(helloworld
        nvshmem::nvshmem
        nvshmem::nvshmem_host
        nvshmem::nvshmem_device)

# Our library
add_library(nvshmem-db SomeLibraryInterfaceFunctions.cu)

include_directories(${CMAKE_CURRENT_DIRECTORY} ${CMAKE_SOURCE_DIR})

target_include_directories(nvshmem-db PUBLIC ${CUDA_INCLUDE_DIRS})

set_property(TARGET nvshmem-db PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(nvshmem-db PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# declare public interface of our library
set_target_properties(nvshmem-db PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/inc/SomeLibraryInterfaceFunctions.h)

if (DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -DDEBUG")
    target_compile_options(nvshmem-db PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-src-in-ptx;-lineinfo>")
    target_compile_definitions(nvshmem-db PRIVATE -DDEBUG)
endif ()
