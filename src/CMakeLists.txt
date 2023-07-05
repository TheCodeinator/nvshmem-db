############################################################
# Our library
############################################################
add_library(nvshmem-db Shuffle.cu)
add_library(baseline-db BaselineShuffle.cu)

target_link_libraries(nvshmem-db PRIVATE nvshmem)

set_property(TARGET nvshmem-db PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(nvshmem-db PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# declare public interface of our library
set_target_properties(nvshmem-db PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/inc/Shuffle.h)
set_target_properties(baseline-db PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/inc/Shuffle.h)
