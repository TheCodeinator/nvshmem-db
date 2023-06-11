#include "SomeLibraryInterfaceFunctions.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

int returnsFour() {
    return 4;
}

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
 * utility function to get the size of a histogram with a particular number of PEs
 */
__host__ __device__ inline int histSizeBytes(const int nPes) {
    return nPes * sizeof(int);
}

// distribution function mapping keys to node IDs
__device__ void distribute(const int key, const int nPes, int &dest) {
    dest = key % nPes;
}

__device__ void histLocalAtomic(const char *const localData,
                                const uint16_t tupleSize,
                                const uint64_t tupleCount,
                                const uint8_t keyOffset,
                                const int nPes,
                                int *const hist) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i{tid};
         i < tupleCount;
         i += blockDim.x * gridDim.x) {

        // assuming a 4-byte integer key
        // get the key of the i-th tuple
        int key = reinterpret_cast<const int *>(localData)[i * tupleSize + keyOffset];
        int dest{0};
        distribute(key, nPes, dest);

        // increment corresponding index in computeOffsets
        atomicAdd(hist + dest, 1);
    }
}

__device__ void histGlobalColl(const int nPes,
                               const nvshmem_team_t team,
                               int *const hist) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) { // only communicate with one thread
        // compute sum of all PE's histograms
        nvshmem_int_sum_reduce(team, hist, hist, nPes);
    }
}


/**
 * exchanges all histograms, such that histograms then contains the histograms of all PEs starting from PE 0, ranging to PE n-1
 * @param histograms pointer to symmetric memory where to put all histograms
 * @param myHistogram pointer to symmetric memory where this histogram is located
 */
__device__ void exchangeHistograms(int *const histograms,
                                   int *const myHistogram,
                                   const nvshmem_team_t &team,
                                   const int nPes) {
    assert(nvshmem_int_alltoall(team, histograms, myHistogram, nPes) == 0);
}

/**
 * Computes a write-offset for each destination PE based on the histograms of each PE.
 * For example, with 4 nodes (zero-indexed), node 3 needs to know the following offsets for its send operations:
  offset(3->0) := 0->0 + 1->0 + 2->0
  offset(3->1) := 0->1 + 1->1 + 2->1
  offset(3->2) := 0->2 + 1->2 + 2->2
  offset(3->3) := 0->3 + 1->3 + 2->3
 */
__device__ void offsetsFromHistograms(const int nPes,
                                      const int thisPe,
                                      const int *const histograms,
                                      int *const offsets) {
    // TODO: parallelize using GPU threads
    if (threadIdx.x == 0) {
        for (int dest{0}; dest < nPes; ++dest) {     // offset for each destination
            for (int pe{0}; pe < thisPe; ++pe) {
                // for each PE get the number of tuples for this destination stored in the histogram
                const int histStart = pe * nPes;
                offsets[dest] += histograms[histStart + dest];
            }
        }
    }
}

/**
 * returns the maximum size of a destination partition based on histograms of all PEs,
 * which is the maximum sum of all tuples that all PEs send to a destination.
 */
__device__ int maxPartitionSize(const int *const histograms, const int nPes) {
    // TODO: parallelize using GPU threads
    int max = 0;

    for (int dest{0}; dest < nPes; ++dest) { // for each destination
        // compute sum of tuples going to this destination
        int sumOfTuples = 0;
        for (int pe{0}; pe < nPes; ++pe) { // for each pe sending to the destination
            const int histStart = pe * nPes;
            sumOfTuples += histograms[histStart + dest];
        }
        if (sumOfTuples > max) {
            max = sumOfTuples;
        }
    }

    return max;
}

__device__ int thisPartitionSize(const int *const histograms, const int nPes, const int thisPe) {
    // TODO: parallelize using GPU threads
    int size = 0;
    for (int pe{0}; pe < nPes; ++pe) {
        const int histStart = pe * nPes;
        size += histograms[histStart + thisPe]; // add the num of tuples each PE has for this PE
    }
}

struct ComputeOffsetsResult {
    int maxPartitionSize;
    int thisPartitionSize;
};

/**
* Computes a global computeOffsets of the data based on the shuffling destination over all nodes.
* Puts the computeOffsets into the device memory pointer hist.
* Returns the maximum number of tuples per node, i.e. the maximum value in the computeOffsets to the host code via
* the device variable intResult.
 * @param localData device memory pointer to data local to this node
 * @param tupleSize size of one tuple
 * @param tupleCount number of tuples
 * @param keyOffset position where to find the 4-byte integer key
 * @param team team used for nvshmem communication
 * @param nPes number of members in the team
 * @param hist device memory pointer to an int array of size nPes to store the computeOffsets
 */
__global__ void computeOffsets(
        char *localData,
        const uint16_t tupleSize,
        const uint64_t tupleCount,
        const uint8_t keyOffset,
        const nvshmem_team_t team,
        const int nPes,
        const int thisPe,
        int *const histograms,
        int *const offsets,
        ComputeOffsetsResult *offsetsResult) {
    // local histogram will be part of the array. histograms are ordered by PE ID. Each histogram has nPes elements
    int *const myHistogram = histograms + (thisPe * nPes);

    // compute local histogram for this PE
    histLocalAtomic(localData, tupleSize, tupleCount, keyOffset, nPes, myHistogram);

    // exchange histograms with other PEs
    exchangeHistograms(histograms, myHistogram, team, nPes);

    // compute offsets based on histograms
    offsetsFromHistograms(nPes, thisPe, histograms, offsets);

    // compute max output partition size (max value of histogram sums)
    offsetsResult->maxPartitionSize = maxPartitionSize(histograms, nPes);

    // compute size of this partition
    offsetsResult->thisPartitionSize = thisPartitionSize(histograms, nPes, thisPe);
}

__global__ void shuffleWithHist(const char *const localData,
                                const uint16_t tupleSize,
                                const uint64_t tupleCount,
                                const uint8_t keyOffset,
                                const nvshmem_team_t team,
                                const int nPes,
                                const int *const hist) {
    // compute shuffle based on offsets

}

// TODO: use tree-based sum reduction for computing the histograms and the sums if possible:
//  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//  or: https://www.cs.ucr.edu/~amazl001/teaching/cs147/S21/slides/13-Histogram.pdf
//  or: https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/histogram64/doc/histogram.pdf

__host__ void shuffle(
        const cudaStream_t &stream,
        const char *const localData, // ptr to device data
        uint16_t tupleSize,
        uint64_t tupleCount,
        uint8_t keyOffset,
        nvshmem_team_t team) {

    int nPes = nvshmem_team_n_pes(team);
    int thisPe = nvshmem_team_my_pe(team);

    // allocate symm. memory for the histograms of all PEs
    // Each histogram contains nPes int elements. There is one computeOffsets for each of the nPes PEs
    int *histograms = static_cast<int *>(nvshmem_malloc(nPes * histSizeBytes(nPes)));

    // allocate private device memory for storing the write offsets;
    // One write offset for each destination PE
    int *offsets;
    cudaMalloc(&offsets, nPes * sizeof(int));

    // allocate private device memory for the result of the computeOffsets function
    ComputeOffsetsResult offsetsResult{};
    ComputeOffsetsResult *offsetsResultDevice;
    cudaMalloc(&offsetsResultDevice, sizeof(ComputeOffsetsResult));

    void *args[] = {const_cast<char **>(&localData), &tupleSize, &tupleCount, &keyOffset, &team, &nPes,
                    &thisPe, &histograms, &offsets, &offsetsResultDevice};
    dim3 dimBlock(1); // TODO: adjust dimensions
    dim3 dimGrid(1);  // TODO: adjust dimensions

    // TODO: need to set sharedMem in launch?
    // compute and exchange the histograms and compute the offsets for remote writing
    nvshmemx_collective_launch((const void *) computeOffsets, dimGrid, dimBlock, args, 0, 0);
    cudaDeviceSynchronize(); // wait for kernel to finish and deliver result

    // get result from kernel launch
    cudaMemcpy(&offsetsResult, offsetsResultDevice, sizeof(ComputeOffsetsResult), cudaMemcpyDeviceToHost);

    // histograms no longer required since offsets have been calculated => release corresponding symm. memory
    nvshmem_free(histograms);

    // allocate symmetric memory big enough to fit the largest partition
    char *symmMem = static_cast<char *>(nvshmem_malloc(offsetsResult.maxPartitionSize * tupleSize));

    // TODO: launch new kernel with the computeOffsets (still in device mem) and the allocated symmetric memory
    //  That kernel then should have everything at hand to do the data shuffling using nvshmem put and non-overlapping offsets

}