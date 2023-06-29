#ifndef NVSHMEM_DB_SHUFFLE_H
#define NVSHMEM_DB_SHUFFLE_H

#include <cuda.h>
#include "nvshmem.h"

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

// used to check the status code of NVSHMEM routines for errors
#define NVSHMEM_CHECK(stmt)                                                                \
    do {                                                                                   \
        int _CHECK_result = (stmt);                                                               \
        if (NVSHMEMX_SUCCESS != _CHECK_result) {                                                  \
            fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, \
                    _CHECK_result);                                                               \
            exit(-1);                                                                      \
        }                                                                                  \
    } while (0)

struct ShuffleResult {
    uint8_t *tuples;
    uint64_t partitionSize;
};

__host__ ShuffleResult shuffle(
    const uint8_t *localData,// ptr to device data
    uint16_t tupleSize,
    uint64_t tupleCount,
    uint8_t keyOffset,
    const cudaStream_t &stream,
    nvshmem_team_t team);

#endif
