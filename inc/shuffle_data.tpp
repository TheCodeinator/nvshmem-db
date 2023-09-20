#ifndef NVSHMEM_DB_SHUFFLEBUFFERS_H
#define NVSHMEM_DB_SHUFFLEBUFFERS_H

#include <cstdint>
#include <cassert>
#include <cuda.h>

#include "NVSHMEMUtils.cuh"

template<typename key_type, typename data_type>
struct Tuple{
    key_type key;
    data_type data;
};

template<typename Tuple>
struct ShuffleData {
    ShuffleData *host_data = nullptr;
    ShuffleData *device_data = nullptr;

    const Tuple *const device_tuples;

    const uint32_t pe_count;

    const uint32_t grid_dim;
    const uint32_t block_dim;
    const uint32_t thread_count;

    const uint64_t tuple_count;
    const uint32_t tuple_size;

    const uint32_t send_buffer_size_in_tuples;
    const uint32_t send_buffer_size_in_bytes;
    const bool allocate_send_buffer;

    const uint32_t batch_count;

    __host__ ShuffleData(const Tuple *device_tuples, uint32_t pe_count, uint32_t grid_dim, uint32_t block_dim,
                         uint64_t tuple_count, uint32_t send_buffer_size_multiplier, bool allocate_send_buffer) :
            host_data(this),
            device_tuples(device_tuples),
            pe_count(pe_count),
            grid_dim(grid_dim),
            block_dim(block_dim),
            thread_count(grid_dim * block_dim),
            tuple_count(tuple_count),
            tuple_size(sizeof(Tuple)),
            send_buffer_size_in_tuples(block_dim * send_buffer_size_multiplier),
            send_buffer_size_in_bytes(send_buffer_size_in_tuples * tuple_size),
            allocate_send_buffer(allocate_send_buffer),
            batch_count(ceil(static_cast<double>(tuple_count) / send_buffer_size_in_tuples))
    {
        assert(tuple_size >= 8);
        assert(tuple_size % 8 == 0);

        CUDA_CHECK(cudaMalloc(&device_data, sizeof(ShuffleData)));
        CUDA_CHECK(cudaMemcpy(device_data, this, sizeof(ShuffleData), cudaMemcpyHostToDevice));
    }
    __host__ __device__ ShuffleData(const ShuffleData &other) = delete;
    __host__ __device__ ShuffleData(ShuffleData &&other) = delete;
    __host__ ~ShuffleData() {
        CUDA_CHECK(cudaFree(device_data));
    }

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
template<typename Tuple>
class SendBuffers {
public:
    SendBuffers *host_buffers = nullptr;
    SendBuffers *device_buffers = nullptr;

    const uint32_t buffer_count = 2;

private:
    uint32_t buffer_in_use = 0;

    const ShuffleData<Tuple> *device_data = nullptr;

    Tuple *buffers = nullptr;
    uint32_t *offsets = nullptr;

public:
    __host__ explicit SendBuffers(const ShuffleData<Tuple> *host_data) :
            host_buffers(this),
            device_data(host_data->device_data)
    {
        if(host_data->allocate_send_buffer)
            buffers = static_cast<Tuple*>(nvshmem_malloc(buffer_count * host_data->grid_dim * host_data->send_buffer_size_in_tuples * sizeof(Tuple) * host_data->pe_count));

        uint32_t offset_buffer_size = buffer_count * host_data->pe_count * host_data->grid_dim * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&offsets, offset_buffer_size));
        CUDA_CHECK(cudaMemset(offsets, 0, offset_buffer_size));

        CUDA_CHECK(cudaMalloc(&device_buffers, sizeof(SendBuffers)));
        CUDA_CHECK(cudaMemcpy(device_buffers, this, sizeof(SendBuffers), cudaMemcpyHostToDevice));
    }
    __host__ __device__ SendBuffers(const SendBuffers &other) = delete;
    __host__ __device__ SendBuffers(SendBuffers &&other) = delete;
    __host__ ~SendBuffers() {
        nvshmem_free(buffers);
        CUDA_CHECK(cudaFree(offsets));
        //CUDA_CHECK(cudaFree(buffer_in_use));
        CUDA_CHECK(cudaFree(device_buffers));
    }

    __host__ __device__ SendBuffers &operator=(const SendBuffers &other) = delete;
    __host__ __device__ SendBuffers &operator=(SendBuffers &&other) = delete;

    __device__ uint32_t currentBufferIndex() const { return buffer_in_use; }

    __device__ Tuple *getBuffer(uint32_t bufferIndex) {
        return buffers + bufferIndex * device_data->send_buffer_size_in_tuples * device_data->pe_count;
    }
    __device__ Tuple *currentBuffer() { return getBuffer(buffer_in_use); }

    __device__ uint32_t *getOffsets(uint32_t bufferIndex) {
        return offsets + bufferIndex * device_data->pe_count;
    }
    __device__ uint32_t *currentOffsets() { return getOffsets(buffer_in_use); }

    /**
     * increase the Buffer index by one (or reset to 0 if last buffer reached) and return the old buffer index
     * @return the buffer index that was used before this call
     */
    __device__ uint32_t useNextBuffer() {
        uint32_t old_buffer_in_use = buffer_in_use;
        buffer_in_use = (old_buffer_in_use + 1) % buffer_count;
        return old_buffer_in_use;
    }
    /**
     * Resets the offsets of the given bufferIndex to 0
     */
    __device__ void resetBuffer(uint bufferIndex) {
        memset(getOffsets(bufferIndex), 0, device_data->pe_count * sizeof(uint32_t));
    }
};


template<typename Tuple>
class ThreadOffsets {
public:
    ThreadOffsets *host_offsets = nullptr;
    ThreadOffsets *device_offsets = nullptr;

private:
    ShuffleData<Tuple> *device_data = nullptr;

    uint32_t *offsets = nullptr;

public:
    __host__ explicit ThreadOffsets(const ShuffleData<Tuple> *host_data)  :
            device_data(host_data->device_data)
    {
        const auto offsets_size = host_data->batch_count * host_data->pe_count * host_data->thread_count * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&offsets, offsets_size));
        CUDA_CHECK(cudaMemset(offsets, 0, offsets_size));

        CUDA_CHECK(cudaMalloc(&device_offsets, sizeof(ThreadOffsets)));
        CUDA_CHECK(cudaMemcpy(device_offsets, this, sizeof(ThreadOffsets), cudaMemcpyHostToDevice));
    }
    __host__ __device__ ThreadOffsets(const ThreadOffsets &other) = delete;
    __host__ __device__ ThreadOffsets(ThreadOffsets &&other) = delete;
    __host__ ~ThreadOffsets() {
        CUDA_CHECK(cudaFree(offsets));
        CUDA_CHECK(cudaFree(device_offsets));
    }

    __host__ __device__ ThreadOffsets &operator=(const ThreadOffsets &other) = delete;
    __host__ __device__ ThreadOffsets &operator=(ThreadOffsets &&other) = delete;

    __device__ uint32_t *getOffset(uint32_t batch, uint32_t thread, uint32_t pe) {
        auto batch_offset = batch * device_data->pe_count * device_data->thread_count;
        auto thread_offset = thread * device_data->pe_count;
        assert(batch_offset + thread_offset + pe < device_data->batch_count * device_data->pe_count * device_data->thread_count);
        return offsets + batch_offset + thread_offset + pe;
    }
};

#endif //NVSHMEM_DB_SHUFFLEBUFFERS_H
