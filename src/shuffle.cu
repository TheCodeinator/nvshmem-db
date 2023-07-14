#include "shuffle.h"

#include <unistd.h>

#include "send_buffers.h"


/**
 * distribution function mapping keys to node IDs.
 * Currently modulo. Could be anything instead
 */
__device__ uint32_t distribute(const uint8_t *const tuple, const uint32_t keyOffset, const uint32_t nPes) {
    // assuming a 4-byte integer key
    return *reinterpret_cast<const uint32_t *>(tuple + keyOffset) % nPes;
}

__device__ void histLocalAtomic(const uint8_t *const localData,
                                const uint16_t tupleSize,
                                const uint64_t tupleCount,
                                const uint8_t keyOffset,
                                const uint32_t nPes,
                                const uint32_t thisPe,
                                uint32_t *const hist,
                                ThreadOffsets *threadOffsets) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t i = tid; i < tupleCount; i += blockDim.x * gridDim.x) {
        // get pointer to the i-th shuffle_tuple
        const uint8_t *const tuplePtr = localData + (i * tupleSize);
        // get destination PE ID of the shuffle_tuple
        const uint32_t dest = distribute(tuplePtr, keyOffset, nPes);

        // increment corresponding index in compute_offsets
        atomicAdd(hist + dest, 1);
        // increment the count for this thread for the current batch and destination (translate it to the offset later)
        ++(*threadOffsets->getOffset(i / threadOffsets->tuplePerBatch, tid, dest));
    }

    __syncthreads();

    // Translate batch counts to offsets
    if (tid == 0) {
        printf("PE %d: batchCount: %d, threadCount: %d, peCount: %d\n", thisPe, threadOffsets->batchCount, threadOffsets->threadCount, nPes);
        for (uint32_t batch = 0; batch < threadOffsets->batchCount; ++batch) { // TODO: parallelize this
            for (uint32_t pe = 0; pe < nPes; ++pe) { // TODO: parallelize this
                uint32_t currentOffset = 0;
                for (uint32_t thread = 0; thread < threadOffsets->threadCount; ++thread) {
                    uint32_t *offset = threadOffsets->getOffset(batch, thread, pe);
                    uint32_t tmp = *offset;
                    *offset = currentOffset;
                    currentOffset += tmp;
                    printf("PE %d: batch %d, thread %d, pe %d, offset %d, count (tmp) %d\n", thisPe, batch, thread, pe, *offset, tmp);
                }
            }
        }
    }
}

/**
 * Computes a write-offset for each destination PE based on the histograms of each PE.
 * For example, with 4 nodes (zero-indexed), node 3 needs to know the following offsets for its send operations:
  offset(3->0) := 0->0 + 1->0 + 2->0
  offset(3->1) := 0->1 + 1->1 + 2->1
  offset(3->2) := 0->2 + 1->2 + 2->2
  offset(3->3) := 0->3 + 1->3 + 2->3
 */
__device__ void offsetsFromHistograms(const uint32_t nPes,
                                      const uint32_t thisPe,
                                      const uint32_t *const histograms,
                                      uint32_t *const offsets) {
    // TODO: parallelize using GPU threads
    if (threadIdx.x == 0) {
        for (int destPe{0}; destPe < nPes; ++destPe) {     // offset for each destination
            for (int source{0}; source < thisPe; ++source) {
                // for each PE get the number of tuples for this destination stored in the histogram
                const int histStart = source * nPes;
                offsets[destPe] += histograms[histStart + destPe];
            }
        }
    }
}

/**
 * returns the maximum size of a destination partition based on histograms of all PEs,
 * which is the maximum sum of all tuples that all PEs send to a destination.
 */
__device__ uint32_t maxPartitionSize(const uint32_t *const histograms, const uint32_t nPes) {
    // TODO: parallelize using GPU threads
    uint32_t max = 0;

    for (uint32_t dest{0}; dest < nPes; ++dest) { // for each destination
        // compute sum of tuples going to this destination
        uint32_t sumOfTuples = 0;
        for (uint32_t pe{0}; pe < nPes; ++pe) { // for each pe sending to the destination
            const uint32_t histStart = pe * nPes;
            sumOfTuples += histograms[histStart + dest];
        }
        if (sumOfTuples > max) {
            max = sumOfTuples;
        }
    }

    return max;
}

__device__ uint32_t thisPartitionSize(const uint32_t *const histograms, const uint32_t nPes, const uint32_t thisPe) {
    // TODO: parallelize using GPU threads
    uint32_t size = 0;
    for (uint32_t pe{0}; pe < nPes; ++pe) {
        const uint32_t histStart = pe * nPes;
        size += histograms[histStart + thisPe]; // add the num of tuples each PE has for this PE
    }
    return size;
}

struct ComputeOffsetsResult {
    uint32_t maxPartitionSize;
    uint32_t thisPartitionSize;
};

/**
* Computes a global compute_offsets of the data based on the shuffling destination over all nodes.
* Puts the compute_offsets into the device memory pointer hist.
* Returns the maximum number of tuples per node, i.e. the maximum value in the compute_offsets to the host code via
* the device variable intResult.
 * @param localData device memory pointer to data local to this node
 * @param tupleSize size of one shuffle_tuple
 * @param tupleCount number of tuples
 * @param keyOffset position where to find the 4-byte integer key
 * @param team team used for nvshmem communication
 * @param nPes number of members in the team
 * @param hist device memory pointer to an int array of size nPes to store the compute_offsets
 */
__global__ void compute_offsets(
        const uint8_t *const localData,
        const uint16_t tupleSize,
        const uint64_t tupleCount,
        const uint8_t keyOffset,
        const nvshmem_team_t team,
        const int nPes,
        const int thisPe,
        uint32_t *localHistogram,
        uint32_t *const globalHistograms,
        uint32_t *const offsets,
        ComputeOffsetsResult *offsetsResult,
        ThreadOffsets *sendBufferThreadOffsets) {
    // local histogram will be part of the array. globalHistograms are ordered by PE ID. Each histogram has nPes elements

    // compute local histogram for this PE
    histLocalAtomic(localData, tupleSize, tupleCount, keyOffset, nPes, thisPe, localHistogram, sendBufferThreadOffsets);

    // TODO: alltoall doesn't work, but fcollect does?
    assert(nvshmem_uint32_fcollect(team, globalHistograms, localHistogram, nPes) == 0);

    if (threadIdx.x == 0) {
        printf("PE %d globalHistograms AFTER exchange: %d %d %d %d\n", thisPe, globalHistograms[0], globalHistograms[1],
               globalHistograms[2], globalHistograms[3]);
    }

    // compute offsets based on globalHistograms
    offsetsFromHistograms(nPes, thisPe, globalHistograms, offsets);

    // print offsets
    if (threadIdx.x == 0) {
        printf("(Remote) offsets for PE %d: ", thisPe);
        for (int i{0}; i < nPes; ++i) {
            printf("%d ", offsets[i]);
        }
        printf("\n");
    }

    // compute max output partition size (max value of histogram sums)
    offsetsResult->maxPartitionSize = maxPartitionSize(globalHistograms, nPes);

    // compute size of this partition
    offsetsResult->thisPartitionSize = thisPartitionSize(globalHistograms, nPes, thisPe);

    printf("PE %d: maxPartitionSize = %d, thisPartitionSize = %d\n", thisPe, offsetsResult->maxPartitionSize,
           offsetsResult->thisPartitionSize);
}

/**
 * Swaps the buffer pointers and position pointers and clears the main buffers
 * Then asynchronously sends results out based on the backup buffers
 * When this function returns, the main buffer and position can be reused (since they have been swapped)
 * This function completes immediately but the networking has to be awaited with nvshmem_quiet before the next call
 * to this function
 */
__device__ void
async_send_data(const uint32_t *offsets, uint8_t *symmMem, SendBuffers *buffers, uint32_t *const positionsRemote) {
    // send data out
    for (int dest{0}; dest < buffers->nPes; ++dest) {
        // send data to remote PE
        nvshmem_uint8_put_nbi(symmMem + (offsets[dest] * buffers->tupleSize +
                                         positionsRemote[dest] * buffers->tupleSize), // position to write in shared mem
                              buffers->currentBuffer() + (dest * buffers->bufferSize), // start position of one destination send buffer
                              buffers->currentOffsets()[dest] * buffers->tupleSize, // number of bytes to send
                              dest); // destination pe
        // advance the position pointer for remote writing for this PE
        positionsRemote[dest] += buffers->currentOffsets()[dest];
    }
    buffers->useNextBuffer();
    buffers->resetBuffer(buffers->currentBufferIndex());
}

__global__ void shuffle_with_offset(const uint8_t *const localData,
                                    const uint64_t tupleCount,
                                    const uint8_t keyOffset,
                                    const nvshmem_team_t team,
                                    const uint32_t thisPe,
                                    const uint32_t *const offsets,
                                    uint8_t *const symmMem,
                                    SendBuffers *buffers,
                                    ThreadOffsets *threadOffsets) {
    assert(buffers->bufferSize % buffers->tupleSize == 0); // buffer size must be a multiple of tuple size

    const uint threadCount = blockDim.x * gridDim.x;
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    // current position writing to the remote locations
    auto positionsRemote = new uint32_t[buffers->nPes];

    uint32_t bufferTupleCount = buffers->bufferSize / buffers->tupleSize;
    const uint iterationToSend = bufferTupleCount / threadCount;
    uint iteration = 0;

    if (tid == 0) {
        printf("PE %d: %d threads, buffer size in tuple %d\n", thisPe, threadCount, bufferTupleCount);
    }

    // iterate over all local data and compute the destination
    for (uint i = tid; i < tupleCount; i += blockDim.x * gridDim.x) {
        // pointer to i-th local tuple
        const uint8_t *const tuplePtr = localData + (i * buffers->tupleSize);
        // get destination of tuple
        const uint dest = distribute(tuplePtr, keyOffset, buffers->nPes);

        // increment the offset for this destination atomically (atomicAdd returns the value before increment)
        //uint32_t offset = atomicAdd(buffers->currentOffsets() + dest, 1);
        uint32_t &offset = *threadOffsets->getOffset(i / threadOffsets->tuplePerBatch, tid, dest);
        assert(offset < bufferTupleCount); // assert that offset is not out of bounds
        printf("PE %d, Thread %d: writes tuple id %d -> %d at offset %d to buffer %d\n", thisPe, tid,
               reinterpret_cast<uint32_t const *>(tuplePtr)[keyOffset], dest, offset, buffers->currentBufferIndex());
        // copy tuple to buffer
        memcpy(buffers->currentBuffer() + (dest * buffers->bufferSize + offset * buffers->tupleSize), // to dest-th buffer with offset position
               tuplePtr,
               buffers->tupleSize);
        offset += 1; // increase offset of thread for next tuple in this thread

        // if there might be a full buffer or tuple count reached, send data out
        // We do not track all buffersComp individually, because we can only await all async operations at once anyways
        //printf("PE: %d, offset: %d\n", thisPe, offset);
        if (++iteration % iterationToSend == 0 || i + (threadCount - tid) >= tupleCount) {
            __syncthreads(); // sync threads before send operation (to ensure that all threads have written their data)
            if (tid == 0) {
                // set offsets for all PEs
                for(uint32_t pe = 0; pe < buffers->nPes; ++pe) {
                    buffers->currentOffsets()[pe] = *threadOffsets->getOffset(i / threadOffsets->tuplePerBatch, threadCount-1, pe);
                }

                // wait for completion of previous send
                nvshmem_quiet();

                // TODO: check if buffer is really full and introduce counter for iteration count left
                // wait for previous send to be completed => buffersBackup reusable after quiet finishes
                printf("PE %d, Thread %d: sending buffer %d\n", thisPe, tid, buffers->currentBufferIndex());
                // send data out (at old buffer)
                async_send_data(offsets, symmMem, buffers, positionsRemote);
            }
            __syncthreads(); // sync threads after send operation
        }
    }
}

// TODO: use tree-based sum reduction for computing the histograms and the sums if possible:
//  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//  or: https://www.cs.ucr.edu/~amazl001/teaching/cs147/S21/slides/13-Histogram.pdf
//  or: https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/histogram64/doc/histogram.pdf


__global__ void print_tuple_result(const uint32_t thisPe, const uint8_t *const data, const uint16_t tupleSize,
                                   const uint64_t tupleCount) {
    // print only the ID of the tuple
    printf("PE %d result (%lu tuples): ", thisPe, tupleCount);
    for (uint64_t i{0}; i < tupleCount; ++i) {
        printf("%lu ", reinterpret_cast<const uint64_t *>(data)[i * 8]);
    }
    printf("\n");
}

/**
 * @param localData pointer to GPU memory where the local partition before shuffling resides
 * @param shuffledData pointer for returning
 * @param tupleSize
 * @param tupleCount
 * @param keyOffset
 * @param stream
 * @param team
 * @return
 */
__host__ ShuffleResult shuffle(
        const uint8_t *const localData, // ptr to device data
        uint16_t tupleSize,
        uint64_t tupleCount,
        uint8_t keyOffset,
        const cudaStream_t &stream,
        nvshmem_team_t team) {

    int nPes = nvshmem_team_n_pes(team);
    int thisPe = nvshmem_team_my_pe(team);
    size_t globalHistogramsSize = nPes * nPes * sizeof(uint32_t);

    printf("PE %d: shuffle with tupleSize = %d, tupleCount = %lu, keyOffset = %d\n", thisPe, tupleSize, tupleCount,
           keyOffset);

    const uint32_t threadCount = 3;
    const uint32_t sendBufferTupleCount = 2 * threadCount; // buffer size must be a multiple of thread count
    assert(threadCount <= sendBufferTupleCount && sendBufferTupleCount % threadCount == 0); // buffer size must be a multiple of thread count for buffer maybe full asumption to send out data

    // allocate symm. memory for the globalHistograms of all PEs
    // Each histogram contains nPes int elements. There is one compute_offsets for each of the nPes PEs
    auto *globalHistograms = static_cast<uint32_t *>(nvshmem_malloc(globalHistogramsSize));

    // allocate private device memory for storing the write offsets
    // this is sent to all other PEs
    auto *localHistogram = static_cast<uint32_t *>(nvshmem_malloc(nPes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(localHistogram, 0, nPes * sizeof(uint32_t)));

    // allocate private device memory for storing the write offsets;
    // One write offset for each destination PE
    uint32_t *offsets;
    CUDA_CHECK(cudaMalloc(&offsets, nPes * sizeof(uint32_t)));

    // allocate private device memory for the result of the compute_offsets function
    ComputeOffsetsResult offsetsResult{};
    ComputeOffsetsResult *offsetsResultDevice;
    CUDA_CHECK(cudaMalloc(&offsetsResultDevice, sizeof(ComputeOffsetsResult)));

    // allocate private device memory for offsets in the send buffer per thread, per batch
    auto hostThreadOffsets = ThreadOffsets(nPes, sendBufferTupleCount, tupleCount, threadCount);
    SendBuffers *deviceThreadOffsets = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceThreadOffsets, sizeof(ThreadOffsets)));
    CUDA_CHECK(cudaMemcpy(deviceThreadOffsets, &hostThreadOffsets, sizeof(ThreadOffsets), cudaMemcpyHostToDevice));

    // TODO: What value should the "sharedMem" argument for the collective launch have?
    // compute and exchange the globalHistograms and compute the offsets for remote writing
    void *comp_offset_args[] = {const_cast<uint8_t **>(&localData), &tupleSize, &tupleCount,
                                &keyOffset, &team, &nPes, &thisPe, &localHistogram,
                                &globalHistograms, &offsets, &offsetsResultDevice,
                                &deviceThreadOffsets};
    NVSHMEM_CHECK(nvshmemx_collective_launch((const void *) compute_offsets, 1, threadCount, comp_offset_args, 1024 * 4, stream));

    // wait for kernel to finish and deliver result
    CUDA_CHECK(cudaDeviceSynchronize());

    // get result from kernel launch
    CUDA_CHECK(cudaMemcpy(&offsetsResult, offsetsResultDevice, sizeof(ComputeOffsetsResult), cudaMemcpyDeviceToHost));

    // histograms no longer required since offsets have been calculated => release corresponding symm. memory
    nvshmem_free(globalHistograms);
    nvshmem_free(localHistogram);

    // allocate symmetric memory big enough to fit the largest partition
    printf("PE %d: Allocating: %u bytes of symmetric memory for tuples after shuffle (%d tuples)\n",
           thisPe, offsetsResult.maxPartitionSize * tupleSize, offsetsResult.maxPartitionSize);
    auto *const symmMem = static_cast<uint8_t *>(nvshmem_malloc(offsetsResult.maxPartitionSize * tupleSize));

    // send buffers for asynchronous send operations
    auto hostSendBuffers = SendBuffers(nPes, sendBufferTupleCount, tupleSize);
    SendBuffers *deviceSendBuffers = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceSendBuffers, sizeof(SendBuffers)));
    CUDA_CHECK(cudaMemcpy(deviceSendBuffers, &hostSendBuffers, sizeof(SendBuffers), cudaMemcpyHostToDevice));

    void *shuffleArgs[] = {const_cast<uint8_t **>(&localData), &tupleCount, &keyOffset,
                           &team, &thisPe, &offsets, const_cast<uint8_t **>(&symmMem),
                           &deviceSendBuffers, &deviceThreadOffsets};

    printf("Calling shuffleWithOffsets with %d PEs\n", nPes);
    // execute the shuffle on the GPU
    NVSHMEM_CHECK(nvshmemx_collective_launch((const void *) shuffle_with_offset, 1, threadCount, shuffleArgs, 1024 * 4, stream));
    CUDA_CHECK(cudaDeviceSynchronize()); // wait for kernel to finish and deliver result
    nvshmem_barrier_all();

    // print tuples after shuffle

    usleep(500000 * thisPe);

    print_tuple_result<<<1, 1, 1024 * 4, stream>>>(thisPe, symmMem, tupleSize, offsetsResult.thisPartitionSize);

    ShuffleResult result{};
    result.partitionSize = offsetsResult.thisPartitionSize;
    result.tuples = reinterpret_cast<uint8_t *>(malloc(offsetsResult.thisPartitionSize * tupleSize));
    CUDA_CHECK(cudaMemcpy(result.tuples, symmMem, offsetsResult.thisPartitionSize * tupleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceSendBuffers));
    CUDA_CHECK(cudaFree(deviceThreadOffsets));
    nvshmem_free(symmMem);
    return result;
}
