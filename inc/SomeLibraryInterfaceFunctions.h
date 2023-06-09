#ifndef NVSHMEM_DB_SOMELIBRARYINTERFACEFUNCTIONS_H
#define NVSHMEM_DB_SOMELIBRARYINTERFACEFUNCTIONS_H

#include <cuda.h>
#include "nvshmem.h"
#include "nvshmemx.h"

int returnsFour();

__device__ void shuffle(
        char* localData,
        uint16_t tupleSize,
        uint64_t tupleCount,
        uint8_t keyOffset,
        nvshmem_team_t team,
        int nPes,
        int thisPe
        );

#endif //NVSHMEM_DB_SOMELIBRARYINTERFACEFUNCTIONS_H
