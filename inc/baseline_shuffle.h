#ifndef NVSHMEM_DB_BASELINE_SHUFFLE_H
#define NVSHMEM_DB_BASELINE_SHUFFLE_H


#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "rdma.hpp"

struct ShuffleResult {
    uint8_t *tuples;
    uint64_t partitionSize;
};

// Shuffle without nvshmem_team
ShuffleResult shuffle(
        const uint8_t *localData,// ptr to device data
        uint16_t tupleSize,
        uint64_t tupleCount,
        uint8_t keyOffset,
        std::string ip,
        uint16_t port,
        std::vector<std::tuple<std::string,uint16_t>> peers,
        const cudaStream_t &stream);

#endif
