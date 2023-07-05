#ifndef NVSHMEM_DB_BASELINE_SHUFFLE_H
#define NVSHMEM_DB_BASELINE_SHUFFLE_H


#include <cuda.h>
#include "rdmapp/rdma.hpp"
#include "rdmapp/connection.hpp"

// used to check the status code of cuda routines for errors
#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t _CHECK_result = (stmt);                                              \
        if (cudaSuccess != _CHECK_result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(_CHECK_result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)


struct ShuffleResult {
    uint8_t *tuples;
    uint64_t partitionSize;
};

// Shuffle without nvshmem_team
__host__ ShuffleResult shuffle(
        const uint8_t *localData,// ptr to device data
        uint16_t tupleSize,
        uint64_t tupleCount,
        uint8_t keyOffset,
        const cudaStream_t &stream);

#endif
