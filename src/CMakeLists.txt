############################################################
# Our library
############################################################
add_library(nvshmem-db Shuffle.cu)

target_link_libraries(nvshmem-db PRIVATE nvshmem)


set_property(TARGET nvshmem-db PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(nvshmem-db PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# declare public interface of our library
set_target_properties(nvshmem-db PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/inc/Shuffle.h)

if (DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -DDEBUG")
    target_compile_options(nvshmem-db PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-src-in-ptx;-lineinfo>")
    target_compile_definitions(nvshmem-db PRIVATE -DDEBUG)
endif ()
