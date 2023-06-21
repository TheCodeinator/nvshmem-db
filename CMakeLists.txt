cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

set (CMAKE_CXX_STANDARD 20)

project(nvshmem-db LANGUAGES CXX CUDA)

find_package(CUDAToolkit 12.0)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12)
set(CUDA_INCLUDE_DIR ${CUDA_TOOLKIT_ROOT_DIR}/include)
set(NVSHMEM_PREFIX /opt/nvshmem)
set(NVSHMEM_INCLUDE_DIR ${NVSHMEM_PREFIX}/include)

set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 70)

MESSAGE("env:CUDA_PATH = '$ENV{CUDA_PATH}'")
MESSAGE("env:PATH = '$ENV{PATH}'")
MESSAGE("env:LD_LIBRARY_PATH = '$ENV{LD_LIBRARY_PATH}'")
MESSAGE("env:LIBRARY_PATH = '$ENV{LIBRARY_PATH}'")
MESSAGE("CUDA_INCLUDE_DIR = ${CUDA_INCLUDE_DIR}")
MESSAGE("NVSHMEM_INCLUDE_DIR = ${NVSHMEM_INCLUDE_DIR}")

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

link_directories(${NVSHMEM_PREFIX}/lib)

find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_PREFIX}/lib/cmake/nvshmem)
add_library(nvshmem ALIAS nvshmem::nvshmem)

# add include directory to all subdirectories recursively
include_directories(inc
        ${NVSHMEM_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIR})

add_subdirectory(src)

# Only build tests and benchmarks if this is the top level cmake file i.e. we build this stand alone without user code
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    enable_testing()
    add_subdirectory(test/src)
    add_subdirectory(benchmarks)
endif ()

if (DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -DDEBUG")
    add_compile_options( "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-src-in-ptx;-lineinfo>")
    add_compile_definitions(-DDEBUG)
endif ()
