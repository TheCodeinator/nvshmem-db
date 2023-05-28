cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(nvshmem-db LANGUAGES CXX CUDA)

set(CMAKE_CUDA_HOST_COMPILER /usr/local/cuda-12/bin/nvcc)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12/bin/nvcc)

set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_subdirectory(src)

# Only build tests if this is the top level cmake file i.e. we build this stand alone without user code
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    enable_testing()
    add_subdirectory(test)
endif ()
