############################################################
# Our library
############################################################
add_library(nvshmem-db shuffle.cu)

target_link_libraries(nvshmem-db PRIVATE nvshmem)
target_include_directories(nvshmem-db PUBLIC ${CMAKE_SOURCE_DIR}/inc)

set_property(TARGET nvshmem-db PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(nvshmem-db PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# declare public interface of our library
set_target_properties(nvshmem-db PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/inc/shuffle.h)
