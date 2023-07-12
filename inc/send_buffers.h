#ifndef NVSHMEM_DB_SHUFFLEBUFFERS_H
#define NVSHMEM_DB_SHUFFLEBUFFERS_H

#include "shuffle.h"

/**
 * @brief A class to manage the send buffers used for shuffling data
 *
 * @details
 * This class manages the send buffers used for shuffling data.
 * The buffers are allocated on the as nvshmem symmetric memory and are used to store the data to send to other PEs.
 * The offset buffer is allocated on the device memory and is used to store the offsets of the data in the buffers.
 */
class SendBuffers {
public:
    const uint32_t bufferCount = 2;

    const uint32_t nPes;
    const uint32_t tupleSize;
    const uint32_t bufferSize;

private:
    uint32_t bufferInUse = 0;

    uint8_t *buffers = nullptr;
    uint32_t *offsets = nullptr;

public:
    __host__ SendBuffers(uint32_t nPes, uint32_t bufferTupleCount, uint32_t tupleSize);
    __host__ __device__ SendBuffers(const SendBuffers &other) = delete;
    __host__ __device__ SendBuffers(SendBuffers &&other) = delete;
    __host__ ~SendBuffers();

    __host__ __device__ SendBuffers &operator=(const SendBuffers &other) = delete;
    __host__ __device__ SendBuffers &operator=(SendBuffers &&other) = delete;

    __device__ uint32_t currentBufferIndex() const;

    __device__ uint8_t *getBuffer(uint bufferIndex);
    __device__ uint8_t *currentBuffers();

    __device__ uint32_t *getOffsets(uint bufferIndex);
    __device__ uint32_t *currentOffsets();

    /**
     * increase the Buffer index by one (or reset to 0 if last buffer reached) and return the old buffer index
     * @return the buffer index that was used before this call
     */
    __device__ uint useNextBuffer();
    /**
     * Resets the offsets of the given bufferIndex to 0
     */
    __device__ void resetBuffer(uint bufferIndex);
};


class ThreadOffsets {
public:
    const uint32_t nPes;
    const uint32_t threadCount;
    const uint32_t tuplePerBatch;
    const uint32_t batchCount;

private:
    uint32_t *offsets = nullptr;

public:
    __host__ ThreadOffsets(uint32_t nPes, uint32_t bufferTupleCount, uint32_t tupleCount, uint32_t threadCount);
    __host__ __device__ ThreadOffsets(const ThreadOffsets &other) = delete;
    __host__ __device__ ThreadOffsets(ThreadOffsets &&other) = delete;
    __host__ ~ThreadOffsets();

    __host__ __device__ ThreadOffsets &operator=(const ThreadOffsets &other) = delete;
    __host__ __device__ ThreadOffsets &operator=(ThreadOffsets &&other) = delete;

    __device__ uint32_t *getOffset(uint32_t batch, uint32_t thread, uint32_t pe);
};

#endif //NVSHMEM_DB_SHUFFLEBUFFERS_H
