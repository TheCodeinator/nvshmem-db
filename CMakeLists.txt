cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(nvshmem-db LANGUAGES CXX CUDA)

set(NVSHMEM_PREFIX /opt/nvshmem)
MESSAGE("env:CUDA_PATH = '$ENV{CUDA_PATH}'")
MESSAGE("env:PATH = '$ENV{PATH}'")
MESSAGE("env:LD_LIBRARY_PATH = '$ENV{LD_LIBRARY_PATH}'")

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12)
find_package(CUDAToolkit 12.0 REQUIRED)
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 70)
message("CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
message("CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

MESSAGE("NVSHMEM_PREFIX set to: '${NVSHMEM_PREFIX}'")
find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_PREFIX}/lib/cmake/nvshmem)

add_subdirectory(src)

# Only build tests if this is the top level cmake file i.e. we build this stand alone without user code
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    enable_testing()
    add_subdirectory(test)
endif ()
