#ifndef NVSHMEM_DB_SHUFFLEBUFFERS_H
#define NVSHMEM_DB_SHUFFLEBUFFERS_H

#include <cstdint>
#include <cuda.h>

struct ShuffleData {
    ShuffleData *host_data = nullptr;
    ShuffleData *device_data = nullptr;

    const uint8_t *const device_tuples;

    const uint32_t pe_count;
    const uint32_t thread_count;

    const uint64_t tuple_count;
    const uint32_t tuple_size;
    const uint8_t key_offset;

    const uint32_t send_buffer_size_in_tuples;
    const uint32_t send_buffer_size_in_bytes;
    const bool allocate_send_buffer;

    const uint32_t batch_count;

    __host__ ShuffleData(const uint8_t *device_tuples, uint32_t pe_count, uint32_t thread_count,
                         uint64_t tuple_count, uint32_t tuple_size, uint8_t key_offset,
                         uint32_t send_buffer_size_multiplier, bool allocate_send_buffer);
    __host__ __device__ ShuffleData(const ShuffleData &other) = delete;
    __host__ __device__ ShuffleData(ShuffleData &&other) = delete;
    __host__ ~ShuffleData();

    __host__ __device__ ShuffleData& operator=(const ShuffleData &other) = delete;
    __host__ __device__ ShuffleData& operator=(ShuffleData &&other) = delete;
};


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
    SendBuffers *host_buffers = nullptr;
    SendBuffers *device_buffers = nullptr;

    const uint32_t buffer_count = 2;

private:
    uint32_t buffer_in_use = 0;

    const ShuffleData *device_data = nullptr;

    uint8_t *buffers = nullptr;
    uint32_t *offsets = nullptr;

public:
    __host__ explicit SendBuffers(const ShuffleData *host_data);
    __host__ __device__ SendBuffers(const SendBuffers &other) = delete;
    __host__ __device__ SendBuffers(SendBuffers &&other) = delete;
    __host__ ~SendBuffers();

    __host__ __device__ SendBuffers &operator=(const SendBuffers &other) = delete;
    __host__ __device__ SendBuffers &operator=(SendBuffers &&other) = delete;

    __device__ uint32_t currentBufferIndex() const;

    __device__ uint8_t *getBuffer(uint bufferIndex);
    __device__ uint8_t *currentBuffer();

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
    ThreadOffsets *host_offsets = nullptr;
    ThreadOffsets *device_offsets = nullptr;

private:
    ShuffleData *device_data = nullptr;

    uint32_t *offsets = nullptr;

public:
    __host__ explicit ThreadOffsets(const ShuffleData *host_data);
    __host__ __device__ ThreadOffsets(const ThreadOffsets &other) = delete;
    __host__ __device__ ThreadOffsets(ThreadOffsets &&other) = delete;
    __host__ ~ThreadOffsets();

    __host__ __device__ ThreadOffsets &operator=(const ThreadOffsets &other) = delete;
    __host__ __device__ ThreadOffsets &operator=(ThreadOffsets &&other) = delete;

    __device__ uint32_t *getOffset(uint32_t batch, uint32_t thread, uint32_t pe);
};

#endif //NVSHMEM_DB_SHUFFLEBUFFERS_H
