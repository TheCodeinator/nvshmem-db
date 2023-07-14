#include "send_buffers.h"


__host__ SendBuffers::SendBuffers(uint32_t nPes, uint32_t bufferTupleCount, uint32_t tupleSize) :
        nPes(nPes),
        tupleSize(tupleSize),
        bufferSize(bufferTupleCount * tupleSize)
{
    buffers = static_cast<uint8_t*>(nvshmem_malloc(bufferCount * bufferSize * nPes * sizeof(uint8_t)));

    uint offsetBufferSize = bufferCount * nPes * sizeof(uint32_t);
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
    return buffers + bufferIndex * bufferSize * nPes;
}
__device__ uint8_t *SendBuffers::currentBuffer()
{
    return getBuffer(bufferInUse);
}

__device__ uint32_t *SendBuffers::getOffsets(uint bufferIndex)
{
    return offsets + bufferIndex * nPes;
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
    memset(getOffsets(bufferIndex), 0, nPes * sizeof(uint32_t));
}



__host__ ThreadOffsets::ThreadOffsets(uint32_t nPes, uint32_t bufferTupleCount, uint32_t tupleCount, uint32_t threadCount) :
        nPes(nPes),
        threadCount(threadCount),
        tuplePerBatch(bufferTupleCount),
        batchCount(ceil(static_cast<double>(tupleCount) / bufferTupleCount))
{
    CUDA_CHECK(cudaMalloc(&offsets, batchCount * nPes * threadCount));
}
__host__ ThreadOffsets::~ThreadOffsets()
{
    CUDA_CHECK(cudaFree(offsets));
}

__device__ uint32_t *ThreadOffsets::getOffset(uint32_t batch, uint32_t thread, uint32_t pe)
{
    auto batchOffset = batch * nPes * threadCount;
    auto threadOffset = thread * nPes;
    return offsets + batchOffset + threadOffset + pe;
}
