#include "shuffle.h"

#include <unistd.h>

#include "shuffle_data.h"

enum class OffsetMode {
    ATOMIC_INCREMENT,
    SYNC_FREE
};

enum class SendBufferMode {
    USE_BUFFER,
    NO_BUFFER
};

__device__ uint getGlobalIdx_3D_3D() {
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x
                   + gridDim.x * gridDim.y * blockIdx.z;
    uint threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                    + (threadIdx.z * (blockDim.x * blockDim.y))
                    + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

/**
 * distribution function mapping keys to node IDs.
 * Currently modulo. Could be anything instead
 */
__device__ inline uint32_t distribute(const uint8_t *const tuple, const uint32_t keyOffset, const uint32_t nPes) {
    // assuming a 4-byte integer key
    return *reinterpret_cast<const uint32_t *>(tuple + keyOffset) % nPes;
}

template <OffsetMode offsetMode>
__device__ void histLocalAtomic(const uint32_t tid,
                                const uint8_t *const localData,
                                const uint32_t thisPe,
                                uint32_t *const hist,
                                ShuffleData *data) {
    for (uint32_t i = tid; i < data->tupleCount; i += data->threadCount) {
        // get pointer to the i-th shuffle_tuple
        const uint8_t *const tuplePtr = localData + (i * data->tupleSize);
        // get destination PE ID of the shuffle_tuple
        const uint32_t dest = distribute(tuplePtr, data->keyOffset, data->peCount);

        // increment corresponding index in compute_offsets
        atomicAdd(hist + dest, 1);
        // increment the count for this thread for the current batch and destination (translate it to the offset later)
        ++(*data->threadOffsets.getOffset(i / data->threadOffsets.tuplePerBatch, tid, dest));
    }

    if constexpr(offsetMode == OffsetMode::SYNC_FREE) {
        __syncthreads();
        for (uint32_t i = tid; i < data->threadOffsets.batchCount * data->peCount; i += data->threadCount) {
            uint32_t batch = i / data->peCount;
            uint32_t dest = i % data->peCount;
            uint32_t currentOffset = 0;
            for (uint32_t thread = 0; thread < data->threadCount; ++thread) {
                uint32_t *offset = data->threadOffsets.getOffset(batch, thread, dest);
                uint32_t tmp = *offset;
                *offset = currentOffset;
                currentOffset += tmp;
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
__device__ void offsetsFromHistograms(const uint32_t tid,
                                      const uint32_t nPes,
                                      const uint32_t thisPe,
                                      const uint32_t *const histograms,
                                      uint32_t *const offsets) {
    // TODO: parallelize using GPU threads
    if (tid == 0) {
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
 * @param hist device memory pointer to an int array of size peCount to store the compute_offsets
 */
template <OffsetMode offsetMode>
__global__ void compute_offsets(
        const nvshmem_team_t team,
        const int thisPe,
        uint32_t *localHistogram,
        uint32_t *const globalHistograms,
        uint32_t *const offsets,
        ComputeOffsetsResult *offsetsResult,
        ShuffleData *data) {

    const uint32_t tid = getGlobalIdx_3D_3D();

    // local histogram will be part of the array. globalHistograms are ordered by PE ID. Each histogram has peCount elements
    // compute local histogram for this PE
    histLocalAtomic<offsetMode>(tid, data->tuples, thisPe, localHistogram, data);

    // TODO: alltoall doesn't work, but fcollect does?
    assert(nvshmem_uint32_fcollect(team, globalHistograms, localHistogram, data->peCount) == 0);

#ifndef NDEBUG
    if (tid == 0) {
        printf("PE %d globalHistograms AFTER exchange: %d %d %d %d\n", thisPe, globalHistograms[0], globalHistograms[1],
               globalHistograms[2], globalHistograms[3]);
    }
#endif

    // compute offsets based on globalHistograms
    offsetsFromHistograms(tid, data->peCount, thisPe, globalHistograms, offsets);

    // print offsets
    if (tid == 0) {
        printf("(Remote) offsets for PE %d: ", thisPe);
        for (int i{0}; i < data->peCount; ++i) {
            printf("%d ", offsets[i]);
        }
        printf("\n");
    }

    // compute max output partition size (max value of histogram sums)
    offsetsResult->maxPartitionSize = maxPartitionSize(globalHistograms, data->peCount);

    // compute size of this partition
    offsetsResult->thisPartitionSize = thisPartitionSize(globalHistograms, data->peCount, thisPe);

    if (tid == 0) {
        printf("PE %d: maxPartitionSize = %d, thisPartitionSize = %d\n", thisPe, offsetsResult->maxPartitionSize,
               offsetsResult->thisPartitionSize);
    }
}


/**
 * Swaps the buffer pointers and position pointers and clears the main buffers
 * Then asynchronously sends results out based on the backup buffers
 * When this function returns, the main buffer and position can be reused (since they have been swapped)
 * This function completes immediately but the networking has to be awaited with nvshmem_quiet before the next call
 * to this function
 */
template <OffsetMode offsetMode>
__device__ void async_send_buffers(uint32_t thisPe, uint32_t threadId, uint32_t tupleId, const uint32_t *offsets,
                                   uint8_t *symmMem, uint32_t *const positionsRemote, ShuffleData *data) {
    // send data out
    for (uint32_t pe = threadId; pe < data->peCount; pe += data->threadCount) {
        uint32_t sendTupleCount;
        if constexpr (offsetMode == OffsetMode::SYNC_FREE) {
            sendTupleCount = *data->threadOffsets.getOffset(tupleId / data->threadOffsets.tuplePerBatch, data->threadCount-1, pe);
        } else if constexpr (offsetMode == OffsetMode::ATOMIC_INCREMENT) {
            sendTupleCount = data->sendBuffers.currentOffsets()[pe];
        }

#ifndef NDEBUG
        printf("PE %d, Thread %d: sends %d tuples to PE %d at offset %d\n",
               thisPe, threadId, sendTupleCount, pe, offsets[pe] + positionsRemote[pe]);
#endif

        // send data to remote PE
        nvshmem_uint8_put_nbi(
                symmMem +
                (offsets[pe] + positionsRemote[pe]) * data->tupleSize, // position to write in shared mem
                data->sendBuffers.currentBuffer() +
                (pe * data->sendBufferSize), // start position of one destination send buffer
                sendTupleCount * data->tupleSize, // number of bytes to send
                pe); // destination pe
        // increment the position pointer for remote writing for this PE
        positionsRemote[pe] += sendTupleCount;
    }
}

template <OffsetMode offsetMode, SendBufferMode sendBufferMode>
__global__ void shuffle_with_offset(const nvshmem_team_t team,
                                    const uint32_t thisPe,
                                    const uint32_t *const offsets,
                                    uint8_t *const symmMem,
                                    ShuffleData *data) {
    assert(data->sendBufferSize % data->tupleSize == 0); // buffer size must be a multiple of tuple size

    const auto tid = getGlobalIdx_3D_3D();
    // current position writing to the remote locations
    auto positionsRemote = new uint32_t[data->peCount];

    const uint iterationToSend = data->sendBufferSizeInTuples / data->threadCount;
    uint iteration = 0;

#ifndef NDEBUG
    if (tid == 0) {
        printf("PE %d: %d threads, buffer size in tuple %d\n", thisPe, data->threadCount, data->sendBufferSizeInTuples);
    }
#endif

    const uint maxIndex = data->threadCount * (data->tupleCount / data->threadCount + (data->tupleCount % data->threadCount != 0));
    // iterate over all local data and compute the destination
    for (uint i = tid; i < maxIndex; i += data->threadCount) {
        uint32_t offset = 0;

        if (i < data->tupleCount) {
            // pointer to i-th local tuple
            const uint8_t *const tuplePtr = data->tuples + (i * data->tupleSize);
            // get destination of tuple
            const uint dest = distribute(tuplePtr, data->keyOffset, data->peCount);

            if constexpr(offsetMode == OffsetMode::SYNC_FREE) {
                auto threadOffset = data->threadOffsets.getOffset(i / data->threadOffsets.tuplePerBatch, tid, dest);
                offset = *threadOffset;
                *threadOffset += 1;
            } else if constexpr(offsetMode == OffsetMode::ATOMIC_INCREMENT) {
                // increment the offset for this destination atomically (atomicAdd returns the value before increment)
                offset = atomicAdd(data->sendBuffers.currentOffsets() + dest, 1);
            } else {
                assert(false);
            }
            if constexpr(sendBufferMode == SendBufferMode::USE_BUFFER) {
                assert(offset < data->sendBufferSizeInTuples); // assert that offset is not out of bounds
#ifndef NDEBUG
                printf("PE %d, Thread %d: writes tuple id %d -> %d at offset %d to buffer %d\n", thisPe, tid,
                       reinterpret_cast<uint32_t const *>(tuplePtr)[data->keyOffset], dest, offset,
                       data->sendBuffers.currentBufferIndex());
#endif
                // copy tuple to buffer
                memcpy(data->sendBuffers.currentBuffer() +
                       (dest * data->sendBufferSize + offset * data->tupleSize), // to dest-th buffer with offset position
                       tuplePtr,
                       data->tupleSize);
            } else if constexpr(sendBufferMode == SendBufferMode::NO_BUFFER) {
#ifndef NDEBUG
                printf("PE %d, Thread %d: writes tuple id %d -> %d at offset %d\n", thisPe, tid,
                       reinterpret_cast<uint32_t const *>(tuplePtr)[data->keyOffset], dest, offsets[dest] + offset);
#endif
                nvshmem_putmem_nbi(symmMem + (offsets[dest] + offset) * data->tupleSize, // position to write in shared mem
                                   tuplePtr, // tuple to send
                                   data->tupleSize, // number of bytes to send
                                   dest); // destination pe
            } else {
                assert(false);
            }
        }

        // if there might be a full buffer or tuple count reached, send data out
        // We do not track all buffersComp individually, because we can only await all async operations at once anyways
        //printf("PE: %d, offset: %d\n", thisPe, offset);
        if constexpr(sendBufferMode == SendBufferMode::USE_BUFFER) {
            if(++iteration % iterationToSend == 0 || i + (data->threadCount - tid) >= data->tupleCount) {
                nvshmem_quiet(); // wait for previous send to be completed => buffersBackup reusable after quiet finishes
                __syncthreads(); // sync threads before send operation (to ensure that all threads have written their data into the buffer)
                // send data parallelized and asyncronously to all destinations
                async_send_buffers<offsetMode>(thisPe, tid, i, offsets, symmMem, positionsRemote, data);
                __syncthreads();
                if(tid == 0) {
                    data->sendBuffers.useNextBuffer(); // switch to the next buffer
                    data->sendBuffers.resetBuffer(
                            data->sendBuffers.currentBufferIndex()); // reset the offsets of the current buffer
                }
                __syncthreads(); // sync threads after send operation
            }
        } else if constexpr(sendBufferMode == SendBufferMode::NO_BUFFER) {
        } else {
            assert(false);
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

template <int MODE>
int add() {
    if constexpr(MODE == 1) {
        return 1;
    } else {
        return 2;
    }
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
    constexpr auto offsetMode = OffsetMode::SYNC_FREE;
    constexpr auto sendBufferMode = SendBufferMode::NO_BUFFER;
    uint32_t sendBufferSizeMultiplier = 30;

    printf("PE %d: shuffle with tupleSize = %d, tupleCount = %lu, keyOffset = %d\n", thisPe, tupleSize, tupleCount,
           keyOffset);

    const auto blockDimension = dim3(2, 2, 1);
    ShuffleData hostShuffleData(localData, nPes, blockDimension.x*blockDimension.y*blockDimension.z, tupleCount, tupleSize, keyOffset, sendBufferSizeMultiplier);
    if constexpr(sendBufferMode == SendBufferMode::USE_BUFFER) {
        assert(hostShuffleData.threadCount <= hostShuffleData.sendBufferSizeInTuples && hostShuffleData.sendBufferSizeInTuples % hostShuffleData.threadCount == 0); // buffer size must be a multiple of thread count for buffer maybe full asumption to send out data
    }
    ShuffleData *deviceShuffleData = hostShuffleData.copyToDevice();

    // allocate symm. memory for the globalHistograms of all PEs
    // Each histogram contains peCount int elements. There is one compute_offsets for each of the peCount PEs
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

    // TODO: What value should the "sharedMem" argument for the collective launch have?
    // compute and exchange the globalHistograms and compute the offsets for remote writing
    void *comp_offset_args[] = {&team, &thisPe, &localHistogram,
                                &globalHistograms, &offsets, &offsetsResultDevice, &deviceShuffleData};
    NVSHMEM_CHECK(nvshmemx_collective_launch((const void *) compute_offsets<offsetMode>, 1, blockDimension, comp_offset_args, 1024 * 4, stream));

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

    void *shuffleArgs[] = {&team, &thisPe, &offsets,
                           const_cast<uint8_t **>(&symmMem), &deviceShuffleData};

    printf("Calling shuffleWithOffsets with %d PEs\n", nPes);

    if constexpr(sendBufferMode == SendBufferMode::NO_BUFFER) {
        nvshmemx_buffer_register(const_cast<void*>(reinterpret_cast<const void*>(localData)), tupleCount * tupleSize);
    }

    // execute the shuffle on the GPU
    NVSHMEM_CHECK(nvshmemx_collective_launch((const void *) shuffle_with_offset<offsetMode, sendBufferMode>, 1, hostShuffleData.threadCount, shuffleArgs, 1024 * 4, stream));
    CUDA_CHECK(cudaDeviceSynchronize()); // wait for kernel to finish and deliver result
    nvshmem_barrier(team); // wait for all send operations to finish

    if constexpr(sendBufferMode == SendBufferMode::NO_BUFFER) {
        nvshmemx_buffer_unregister(const_cast<void*>(reinterpret_cast<const void*>(localData)));
    }

    for(int i = 0; i < nPes; i++) {
        nvshmem_barrier(team);
        if(i == thisPe) {
            print_tuple_result<<<1, 1, 1024 * 4, stream>>>(thisPe, symmMem, tupleSize, offsetsResult.thisPartitionSize);
            CUDA_CHECK(cudaDeviceSynchronize()); // wait for kernel to finish
        }
        nvshmem_barrier(team);
    }

    ShuffleResult result{};
    result.partitionSize = offsetsResult.thisPartitionSize;
    result.tuples = reinterpret_cast<uint8_t *>(malloc(offsetsResult.thisPartitionSize * tupleSize));
    CUDA_CHECK(cudaMemcpy(result.tuples, symmMem, offsetsResult.thisPartitionSize * tupleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceShuffleData));
    nvshmem_free(symmMem);
    return result;
}
