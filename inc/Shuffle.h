#ifndef NVSHMEM_DB_SHUFFLE_H
#define NVSHMEM_DB_SHUFFLE_H

#include <cuda.h>
#include "nvshmem.h"

struct ShuffleResult {
    char *tuples;
    uint64_t partitionSize;
};

__host__ ShuffleResult shuffle(
        const char *const localData,// ptr to device data
        uint16_t tupleSize,
        uint64_t tupleCount,
        uint8_t keyOffset,
        const cudaStream_t &stream,
        nvshmem_team_t team);

#endif
