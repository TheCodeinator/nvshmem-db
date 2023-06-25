#include "Shuffle.h"
#include <unistd.h>

/**
 * Variable for returning integers from cuda kernels to host code
 */
__device__ int intResult;

__host__ int getIntResult() {
    int ret{};
    cudaMemcpyFromSymbol(&ret, "intResult", sizeof(ret), 0, cudaMemcpyDeviceToHost);
    return ret;
}

/**
 * simple swap function callable from device
 */
template<typename T>
__device__ inline void devSwap(T &t1, T &t2) {
    T tmp{t1};
    t1 = t2;
    t2 = tmp;
}

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
                                uint32_t *const hist) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t i{tid};
         i < tupleCount;
         i += blockDim.x * gridDim.x) {

        // get pointer to the i-th shuffle_tuple
        const uint8_t *const tuplePtr = localData + (i * tupleSize);
        // get destination PE ID of the shuffle_tuple
        const uint32_t dest = distribute(tuplePtr, keyOffset, nPes);

        // increment corresponding index in computeOffsets
        atomicAdd(hist + dest, 1);
    }
}

/**
 * exchanges all globalHistograms, such that globalHistograms then contains the globalHistograms of all PEs starting from PE 0, ranging to PE n-1
 * @param globalHistograms pointer to symmetric memory where to put all globalHistograms
 * @param myHistogram pointer to symmetric memory where this histogram is located
 *
 * Histogram layout is as follows (example with 3 PEs):
 *  [0->0] [0->1] [0->2] [1->0] [1->1] [1->2] [2->0] [2->1] [2->2]
 * | Histogram of PE0   | Histogram of PE1   | Histogram of PE2   |
 */
__device__ void exchangeHistograms(const nvshmem_team_t &team,
                                   uint32_t *globalHistograms,
                                   uint32_t *const myHistogram,
                                   const size_t nPes,
                                   const int thisPe) {

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
* Computes a global computeOffsets of the data based on the shuffling destination over all nodes.
* Puts the computeOffsets into the device memory pointer hist.
* Returns the maximum number of tuples per node, i.e. the maximum value in the computeOffsets to the host code via
* the device variable intResult.
 * @param localData device memory pointer to data local to this node
 * @param tupleSize size of one shuffle_tuple
 * @param tupleCount number of tuples
 * @param keyOffset position where to find the 4-byte integer key
 * @param team team used for nvshmem communication
 * @param nPes number of members in the team
 * @param hist device memory pointer to an int array of size nPes to store the computeOffsets
 */
__global__ void computeOffsets(
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
        ComputeOffsetsResult *offsetsResult) {
    // local histogram will be part of the array. globalHistograms are ordered by PE ID. Each histogram has nPes elements

    // compute local histogram for this PE
    histLocalAtomic(localData, tupleSize, tupleCount, keyOffset, nPes, localHistogram);

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

__device__ void
asyncSendData(const uint16_t tupleSize, const uint32_t nPes, const uint32_t *offsets, uint8_t *symmMem,
              uint8_t *buffersComp,
              uint8_t *buffersBackup, uint32_t *const positionsLocal, uint32_t *const positionsRemote) {
    // swap buffer pointers, backup buffer is free and becomes the new compute buffer
    devSwap(buffersComp, buffersBackup);
    // send data out
    for (uint32_t pe{0}; pe < nPes; ++pe) {
        // send data to remote PE
        nvshmem_uint8_put_nbi(&symmMem[offsets[pe] + positionsRemote[pe]],
                              &buffersBackup[pe],
                              positionsLocal[pe] * tupleSize,
                              pe);
        // advance the position pointer for remote writing for this PE
        positionsRemote[pe] += positionsLocal[pe];
    }

    // reset all local position pointers, since we now start with empty buffers again
    memset(positionsLocal, 0, nPes * sizeof(int));
}

struct ShuffleWithOffsetsResult {
    uint8_t *localPartition; // pointer to symmetric memory where the local partition after shuffling resides
};

__global__ void shuffle_with_offset(const uint8_t *const localData,
                                    const uint16_t tupleSize,
                                    const uint64_t tupleCount,
                                    const uint8_t keyOffset,
                                    const nvshmem_team_t team,
                                    const uint32_t nPes,
                                    const uint32_t thisPe,
                                    const uint32_t *const offsets,
                                    uint8_t *const symmMem) {



    // TODO: parallelize scan using GPU threads

    // as a first stupid implementation, we let one thread do everything
    if (threadIdx.x == 0) {
        printf("PE: %d shuffle_with_offset tupleSize: %d, tupleCount: %lu, offsets: %d %d\n", thisPe, tupleSize,
               tupleCount, offsets[0], offsets[1]);
        // number of tuples per send buffer
        const int bufferMaxTuples = 8;
        // allocation for the send buffers: Two sendBuffer to overlap computatimon with transmission
        auto *sendBuffer = new uint8_t[bufferMaxTuples * tupleSize * nPes]; // 64 * 64 * 2 = 8kB
        // current positions in send sendBuffer
        auto *const bytesBuffered = new uint32_t[nPes];
        // current position writing to the remote locations
//        auto *const positionsRemote = new uint32_t[nPes];

        printf("PE %d: bufferMaxTuples = %d, tupleCount = %lu, nPes = %d\n", thisPe, bufferMaxTuples, tupleCount, nPes);

        // iterate over all local data and compute the destination
        for (uint32_t i{0}; i < tupleCount; ++i) {
            // pointer to i-th local tuple
            const uint8_t *const tuplePtr = localData + (i * tupleSize);
            // get destination of tuple
            const uint32_t targetPe = distribute(tuplePtr, keyOffset, nPes);
            // copy shuffle_tuple to respective output buffer

            printf("PE %d: tuple %d(%lu) goes to PE %d (pos %d)\n",
                   thisPe, i, reinterpret_cast<const uint64_t *>(tuplePtr)[0],
                   targetPe,
                   targetPe * bufferMaxTuples * tupleSize + bytesBuffered[targetPe]);

            memcpy(sendBuffer + (targetPe * bufferMaxTuples * tupleSize +
                                 bytesBuffered[targetPe]), // to targetPe-th buffer with offset position
                   tuplePtr,
                   tupleSize);

            // increment local buffer position for this PE
            bytesBuffered[targetPe] += tupleSize;
        }


        for (uint32_t pe{0}; pe < nPes; ++pe) {
            for (uint32_t i{0}; i < bytesBuffered[pe] / tupleSize; ++i) {
                printf("PE %d -> %d, sendBuffer[%d].id = %lu\n",
                       thisPe, pe, pe * bufferMaxTuples + i,
                        // pe * bufferMaxTuples * 64bit elements in tuple + 8 bytes per element * i-th element
                       reinterpret_cast<const uint64_t *>(sendBuffer)[pe * bufferMaxTuples * 8 + 8 * i]);
            }
        }

        nvshmem_quiet();

        for (uint32_t pe{0}; pe < nPes; ++pe) {
            printf("PE %d: sending %d bytes to PE %d at offset %d to %p\n", thisPe, bytesBuffered[pe], pe,
                   offsets[pe], &symmMem[offsets[pe] * tupleSize]);
            for (uint32_t i{0}; i < bytesBuffered[pe] / tupleSize; ++i) {
                printf("PE %d sending -> %d, sendBuffer[%d].id = %lu\n",
                       thisPe, pe, pe * bufferMaxTuples + i,
                        // pe * bufferMaxTuples * 64bit elements in tuple + 8 bytes per element * i-th element
                       reinterpret_cast<const uint64_t *>(sendBuffer)[pe * bufferMaxTuples * 8 + 8 * i]);
            }
            nvshmem_uint8_put_nbi(&symmMem[offsets[pe] * tupleSize],
                                  &sendBuffer[pe * bufferMaxTuples * tupleSize],
                                  bytesBuffered[pe],
                                  pe);
        }

        nvshmem_quiet();

        for (uint32_t i{0}; i < 7; ++i) { // 7 = maxPartitionSize
            printf("PE %d: symmMem[%d].id = %lu\n",
                   thisPe, i,
                    // pe * bufferMaxTuples * 64bit elements in tuple + 8 bytes per element * i-th element
                   reinterpret_cast<const uint64_t *>(symmMem)[8 * i]);
        }

//        asyncSendData(tupleSize, nPes, offsets, symmMem, sendBuffer, buffersBackup, bytesBuffered,
//                      positionsRemote);

        // wait for completion of previous send
        nvshmem_quiet();
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

    printf("PE: %d shuffle with tupleSize = %d, tupleCount = %lu, keyOffset = %d\n", thisPe, tupleSize, tupleCount,
           keyOffset);

    // allocate symm. memory for the globalHistograms of all PEs
    // Each histogram contains nPes int elements. There is one computeOffsets for each of the nPes PEs
    auto *globalHistograms = static_cast<uint32_t *>(nvshmem_malloc(globalHistogramsSize));

    // allocate private device memory for storing the write offsets
    // this is sent to all other PEs
    auto *localHistogram = static_cast<uint32_t *>(nvshmem_malloc(nPes * sizeof(uint32_t)));

    // allocate private device memory for storing the write offsets;
    // One write offset for each destination PE
    uint32_t *offsets;
    CUDA_CHECK(cudaMalloc(&offsets, nPes * sizeof(uint32_t)));

    // allocate private device memory for the result of the computeOffsets function
    ComputeOffsetsResult offsetsResult{};
    ComputeOffsetsResult *offsetsResultDevice;
    CUDA_CHECK(cudaMalloc(&offsetsResultDevice, sizeof(ComputeOffsetsResult)));

    // TODO: What value should the "sharedMem" argument for the collective launch have?
    // compute and exchange the globalHistograms and compute the offsets for remote writing
    computeOffsets<<<1, 1, 1024 * 4, stream>>>(localData, tupleSize, tupleCount,
                                               keyOffset, team, nPes, thisPe, localHistogram,
                                               globalHistograms, offsets, offsetsResultDevice);
    CUDA_CHECK(cudaDeviceSynchronize()); // wait for kernel to finish and deliver result

    // get result from kernel launch
    CUDA_CHECK(cudaMemcpy(&offsetsResult, offsetsResultDevice, sizeof(ComputeOffsetsResult), cudaMemcpyDeviceToHost));

    // histograms no longer required since offsets have been calculated => release corresponding symm. memory
    nvshmem_free(globalHistograms);
    nvshmem_free(localHistogram);

    // allocate symmetric memory big enough to fit the largest partition
    printf("PE: %d Allocating: %u bytes of symmetric memory for tuples after shuffle (%d tuples)\n",
           thisPe, offsetsResult.maxPartitionSize * tupleSize, offsetsResult.maxPartitionSize);
    auto *const symmMem = static_cast<uint8_t *>(nvshmem_malloc(offsetsResult.maxPartitionSize * tupleSize));

    void *shuffleArgs[] = {const_cast<uint8_t **>(&localData), &tupleSize, &tupleCount, &keyOffset,
                           &team, &nPes, &thisPe, &offsets, const_cast<uint8_t **>(&symmMem)};

    printf("Calling shuffleWithOffsets with %d PEs\n", nPes);
    // execute the shuffle on the GPU
    NVSHMEM_CHECK(nvshmemx_collective_launch((const void *) shuffle_with_offset, 1, 1, shuffleArgs, 1024 * 4, stream));
    CUDA_CHECK(cudaDeviceSynchronize()); // wait for kernel to finish and deliver result

    // print tuples after shuffle

    usleep(500000 * thisPe);

    print_tuple_result<<<1, 1, 1024 * 4, stream>>>(thisPe, symmMem, tupleSize, offsetsResult.thisPartitionSize);

    ShuffleResult result{};
    result.partitionSize = offsetsResult.thisPartitionSize;
    result.tuples = reinterpret_cast<uint8_t *>(malloc(offsetsResult.thisPartitionSize * tupleSize));
    CUDA_CHECK(cudaMemcpy(result.tuples, symmMem, offsetsResult.thisPartitionSize * tupleSize, cudaMemcpyDeviceToHost));
    nvshmem_free(symmMem);
    return result;
}
