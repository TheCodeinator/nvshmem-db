#include "shuffle_data.h"


__host__ SendBuffers::SendBuffers(ShuffleData *data) :
        data(data)
{
    if(data->sendBufferSize > 0)
        buffers = static_cast<uint8_t*>(nvshmem_malloc(bufferCount * data->sendBufferSize * data->peCount * sizeof(uint8_t)));

    uint offsetBufferSize = bufferCount * data->peCount * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&offsets, offsetBufferSize));
    CUDA_CHECK(cudaMemset(offsets, 0, offsetBufferSize));
}
__host__ SendBuffers::~SendBuffers()
{
    nvshmem_free(buffers);
    CUDA_CHECK(cudaFree(offsets));
}

__device__ uint32_t SendBuffers::currentBufferIndex() const
{
    return bufferInUse;
}

__device__ uint8_t *SendBuffers::getBuffer(uint bufferIndex)
{
    return buffers + bufferIndex * data->sendBufferSize * data->peCount;
}
__device__ uint8_t *SendBuffers::currentBuffer()
{
    return getBuffer(bufferInUse);
}

__device__ uint32_t *SendBuffers::getOffsets(uint bufferIndex)
{
    return offsets + bufferIndex * data->peCount;
}
__device__ uint32_t *SendBuffers::currentOffsets()
{
    return getOffsets(bufferInUse);
}

__device__ uint SendBuffers::useNextBuffer()
{
    uint oldBufferInUse = bufferInUse;
    bufferInUse = (bufferInUse + 1) % bufferCount;
    return oldBufferInUse;
}
__device__ void SendBuffers::resetBuffer(uint bufferIndex)
{
    memset(getOffsets(bufferIndex), 0, data->peCount * sizeof(uint32_t));
}



__host__ ThreadOffsets::ThreadOffsets(ShuffleData *data) :
        data(data),
        tuplePerBatch(data->sendBufferSizeInTuples),
        batchCount(ceil(static_cast<double>(data->tupleCount) / data->sendBufferSizeInTuples))
{
    CUDA_CHECK(cudaMalloc(&offsets, batchCount * data->peCount * data->threadCount));
}
__host__ ThreadOffsets::~ThreadOffsets()
{
    CUDA_CHECK(cudaFree(offsets));
}

__device__ uint32_t *ThreadOffsets::getOffset(uint32_t batch, uint32_t thread, uint32_t pe)
{
    auto batchOffset = batch * data->peCount * data->threadCount;
    auto threadOffset = thread * data->peCount;
    return offsets + batchOffset + threadOffset + pe;
}



__host__ ShuffleData::ShuffleData(const uint8_t *const tuples, uint32_t peCount, uint32_t threadCount,
                                  uint64_t tupleCount, uint32_t tupleSize, uint8_t keyOffset,
                                  uint32_t sendBufferSizeMultiplier) :
        tuples(tuples),
        peCount(peCount),
        threadCount(threadCount),
        tupleCount(tupleCount),
        tupleSize(tupleSize),
        keyOffset(keyOffset),
        sendBufferSizeInTuples(threadCount * sendBufferSizeMultiplier),
        sendBufferSize(sendBufferSizeInTuples * tupleSize),
        sendBuffers(SendBuffers(this)),
        threadOffsets(ThreadOffsets(this))
{
    //printf("ShuffleData: peCount=%u, threadCount=%u, tupleCount=%lu, tupleSize=%u, keyOffset=%u, sendBufferSizeMultiplier=%u, sendBuffersizeInTuples=%u, sendBufferSize=%u\n",
    //       peCount, threadCount, tupleCount, tupleSize, keyOffset, sendBufferSizeMultiplier, sendBufferSizeInTuples, sendBufferSize);
}
__host__ ShuffleData::~ShuffleData() = default;

__host__ ShuffleData *ShuffleData::copyToDevice()
{
    ShuffleData *deviceData;
    CUDA_CHECK(cudaMalloc(&deviceData, sizeof(ShuffleData)));
    sendBuffers.data = deviceData;
    threadOffsets.data = deviceData;
    CUDA_CHECK(cudaMemcpy(deviceData, this, sizeof(ShuffleData), cudaMemcpyHostToDevice));
    return deviceData;
}
