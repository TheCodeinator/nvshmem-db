#include "shuffle_data_tmp.h"

#include "NVSHMEMUtils.cuh"

__host__ ShuffleData::ShuffleData(const uint8_t *device_tuples, uint32_t pe_count, uint32_t thread_count, uint64_t tuple_count,
                                  uint32_t tuple_size, uint8_t key_offset, uint32_t send_buffer_size_multiplier,
                                  bool allocate_send_buffer) :
        host_data(this),
        device_tuples(device_tuples),
        pe_count(pe_count),
        thread_count(thread_count),
        tuple_count(tuple_count),
        tuple_size(tuple_size),
        key_offset(key_offset),
        send_buffer_size_in_tuples(thread_count * send_buffer_size_multiplier),
        send_buffer_size_in_bytes(send_buffer_size_in_tuples * tuple_size),
        allocate_send_buffer(allocate_send_buffer),
        batch_count(ceil(static_cast<double>(tuple_count) / send_buffer_size_in_tuples))
{
    assert(tuple_size >= 8);
    assert(tuple_size % 8 == 0);

    CUDA_CHECK(cudaMalloc(&device_data, sizeof(ShuffleData)));
    CUDA_CHECK(cudaMemcpy(device_data, this, sizeof(ShuffleData), cudaMemcpyHostToDevice));
}

__host__ ShuffleData::~ShuffleData()
{
    CUDA_CHECK(cudaFree(device_data));
}



__host__ SendBuffers::SendBuffers(const ShuffleData *host_data) :
        host_buffers(this),
        device_data(host_data->device_data)
{
    if(host_data->allocate_send_buffer)
        buffers = static_cast<uint8_t*>(nvshmem_malloc(buffer_count * host_data->send_buffer_size_in_bytes * host_data->pe_count * sizeof(uint8_t)));

    uint offset_buffer_size = buffer_count * host_data->pe_count * sizeof(uint32_t);

    CUDA_CHECK(cudaMalloc(&offsets, offset_buffer_size));
    CUDA_CHECK(cudaMemset(offsets, 0, offset_buffer_size));

    CUDA_CHECK(cudaMalloc(&device_buffers, sizeof(SendBuffers)));
    CUDA_CHECK(cudaMemcpy(device_buffers, this, sizeof(SendBuffers), cudaMemcpyHostToDevice));
}
__host__ SendBuffers::~SendBuffers()
{
    nvshmem_free(buffers);
    CUDA_CHECK(cudaFree(offsets));
    CUDA_CHECK(cudaFree(device_buffers));
}

__device__ uint32_t SendBuffers::currentBufferIndex() const
{
    return buffer_in_use;
}

__device__ uint8_t *SendBuffers::getBuffer(uint bufferIndex)
{
    return buffers + bufferIndex * device_data->send_buffer_size_in_bytes * device_data->pe_count;
}
__device__ uint8_t *SendBuffers::currentBuffer()
{
    return getBuffer(buffer_in_use);
}

__device__ uint32_t *SendBuffers::getOffsets(uint bufferIndex)
{
    return offsets + bufferIndex * device_data->pe_count;
}
__device__ uint32_t *SendBuffers::currentOffsets()
{
    return getOffsets(buffer_in_use);
}

__device__ uint SendBuffers::useNextBuffer()
{
    uint oldBufferInUse = buffer_in_use;
    buffer_in_use = (buffer_in_use + 1) % buffer_count;
    return oldBufferInUse;
}
__device__ void SendBuffers::resetBuffer(uint bufferIndex)
{
    memset(getOffsets(bufferIndex), 0, device_data->pe_count * sizeof(uint32_t));
}



__host__ ThreadOffsets::ThreadOffsets(const ShuffleData *host_data) :
        device_data(host_data->device_data)
{
    const auto offsets_size = host_data->batch_count * host_data->pe_count * host_data->thread_count * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&offsets, offsets_size));
    CUDA_CHECK(cudaMemset(offsets, 0, offsets_size));

    CUDA_CHECK(cudaMalloc(&device_offsets, sizeof(ThreadOffsets)));
    CUDA_CHECK(cudaMemcpy(device_offsets, this, sizeof(ThreadOffsets), cudaMemcpyHostToDevice));
}
__host__ ThreadOffsets::~ThreadOffsets()
{
    CUDA_CHECK(cudaFree(offsets));
    CUDA_CHECK(cudaFree(device_offsets));
}

__device__ uint32_t *ThreadOffsets::getOffset(uint32_t batch, uint32_t thread, uint32_t pe)
{
    auto batch_offset = batch * device_data->pe_count * device_data->thread_count;
    auto thread_offset = thread * device_data->pe_count;
    assert(batch_offset + thread_offset + pe < device_data->batch_count * device_data->pe_count * device_data->thread_count);
    return offsets + batch_offset + thread_offset + pe;
}
