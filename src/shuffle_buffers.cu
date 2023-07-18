#include "shuffle_buffers.h"


__host__ ShuffleBuffers::ShuffleBuffers(uint32_t bufferTupleSize, uint32_t tupleSize, uint32_t nPes) :
        bufferTupleSize(bufferTupleSize),
        tupleSize(tupleSize),
        bufferSize(bufferTupleSize * tupleSize),
        nPes(nPes)
{
    buffers = static_cast<uint8_t*>(nvshmem_malloc(bufferCount * bufferSize * nPes * sizeof(uint8_t)));

    uint offset_buffer_size = bufferCount * nPes * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&offsets, offset_buffer_size));
    cudaMemset(offsets, 0, offset_buffer_size);
}

__host__ ShuffleBuffers::~ShuffleBuffers()
{
    nvshmem_free(buffers);
    CUDA_CHECK(cudaFree(offsets));
}
