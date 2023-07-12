#ifndef NVSHMEM_DB_SHUFFLEBUFFERS_H
#define NVSHMEM_DB_SHUFFLEBUFFERS_H

#include "shuffle.h"

/**
 * @brief A class to manage the buffers used for shuffling data
 *
 * @details
 * This class manages the buffers used for shuffling data.
 * The buffers are allocated on the as nvshmem symmetric memory and are used to store the data to send to other PEs.
 * The offset buffer is allocated on the device memory and is used to store the offsets of the data in the buffers.
 */
class ShuffleBuffers {

public:
    const uint32_t bufferCount = 2;

    const uint32_t nPes;
    const uint32_t bufferTupleSize;
    const uint32_t tupleSize;
    const uint32_t bufferSize;

private:
    uint32_t bufferInUse = 0;

    uint8_t *buffers = nullptr;
    uint32_t *offsets = nullptr;

public:
    __host__ ShuffleBuffers(uint32_t bufferTupleSize, uint32_t tupleSize, uint32_t nPes);
    __host__ __device__ ShuffleBuffers(const ShuffleBuffers &other) = delete;
    __host__ __device__ ShuffleBuffers(ShuffleBuffers &&other) = delete;
    __host__ ~ShuffleBuffers();

    __host__ __device__ ShuffleBuffers &operator=(const ShuffleBuffers &other) = delete;
    __host__ __device__ ShuffleBuffers &operator=(ShuffleBuffers &&other) = delete;

    __device__ inline uint32_t currentBufferIndex() {
        return bufferInUse;
    }

    __device__ inline uint8_t *getBuffer(uint bufferIndex) {
        return buffers + bufferIndex * bufferSize * nPes;
    }
    __device__ inline uint8_t *currentBuffers() {
        return getBuffer(bufferInUse);
    }

    __device__ inline uint32_t *getOffsets(uint bufferIndex) {
        return offsets + bufferIndex * nPes;
    }
    __device__ inline uint32_t *currentOffsets() {
        return getOffsets(bufferInUse);
    }

    /**
     * increase the Buffer index by one (or reset to 0 if last buffer reached) and return the old buffer index
     * @return the buffer index that was used before this call
     */
    __device__ inline uint useNextBuffer() {
        uint oldBufferInUse = bufferInUse;
        bufferInUse = (bufferInUse + 1) % bufferCount;
        return oldBufferInUse;
    }
    /**
     * Resets the offsets of the given bufferIndex to 0
     */
    __device__ inline void resetBuffer(uint bufferIndex) {
        memset(getOffsets(bufferIndex), 0, nPes * sizeof(uint32_t));
    }
};

#endif //NVSHMEM_DB_SHUFFLEBUFFERS_H
