#ifndef NVSHMEM_DB_SHUFFLE_H
#define NVSHMEM_DB_SHUFFLE_H

#include <cuda.h>
#include "nvshmem.h"

__host__ uint64_t shuffle(
        const char *const localData,// ptr to device data
        char *&shuffledData, // pointer to nothing, this function will allocate on host mem for return value
        uint16_t tupleSize,
        uint64_t tupleCount,
        uint8_t keyOffset,
        const cudaStream_t &stream,
        nvshmem_team_t team);

#endif
