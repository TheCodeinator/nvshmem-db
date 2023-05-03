cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(nvshmem-db LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(src)
