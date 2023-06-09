#include "SomeLibraryInterfaceFunctions.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

int returnsFour() {
    return 4;
}

__device__ int intResult;

struct ShuffleAllocation {
    int *histogram;
    char *symmMem;
};

// distribution function mapping keys to node IDs
__device__ void distribute(int key, int nPes, int &dest) {
    dest = key % nPes;
}

__device__ void histLocalAtomic(char *localData,
                                uint16_t tupleSize,
                                uint64_t tupleCount,
                                uint8_t keyOffset,
                                int nPes,
                                int *hist) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i{tid};
         i < tupleCount;
         i += blockDim.x * gridDim.x) {

        // assuming a 4-byte integer key
        // get the key of the i-th tuple
        int key = reinterpret_cast<int *>(localData)[i * tupleSize + keyOffset];
        int dest{0};
        distribute(key, nPes, dest);

        // increment corresponding index in histogram
        atomicAdd(hist + dest, 1);
    }
}

__device__ void histGlobalColl(int nPes,
                               nvshmem_team_t team,
                               int *hist) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) { // only communicate with one thread
        // compute sum of all PE's histograms
        nvshmem_int_sum_reduce(team, hist, hist, nPes);
    }
}

__device__ int sequentialMax(int *hist, int nPes) {
    int max = hist[0];
    for (int i{1}; i < nPes; ++i) {
        if (hist[i] > max) {
            max = hist[i];
        }
    }
    return max;
}

__global__ void histogram(
        char *localData,
        uint16_t tupleSize,
        uint64_t tupleCount,
        uint8_t keyOffset,
        nvshmem_team_t team,
        int nPes,
        int *hist) {
    // compute local histogram
    histLocalAtomic(localData, tupleSize, tupleCount, keyOffset, nPes, hist);

    // global histogram
    histGlobalColl(nPes, team, hist);

    // TODO: return histogram to caller to be reused in second kernel call

    // return maximum to host via device variable
    if (threadIdx.x == 0) {
        intResult = sequentialMax(hist, nPes);
    }
}

// TODO: use tree-based sum reduction for computing the histograms and the sums if possible:
//  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//  or: https://www.cs.ucr.edu/~amazl001/teaching/cs147/S21/slides/13-Histogram.pdf
//  or: https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/histogram64/doc/histogram.pdf

__host__ void shuffle(
        cudaStream_t &stream,
        char *localData, // ptr to device data
        uint16_t tupleSize,
        uint64_t tupleCount,
        uint8_t keyOffset,
        nvshmem_team_t team,
        int nPes,
        int thisPe) {


    // allocate device memory for the histogram
    ShuffleAllocation alloc;
    cudaMalloc(&alloc.histogram, nPes * sizeof(int));

    // compute the histogram on the device
    histogram<<<1, 1, 0, stream>>>(localData,
                                   tupleSize,
                                   tupleCount,
                                   keyOffset,
                                   team,
                                   nPes,
                                   alloc.histogram);
    int maxTuplesPerNode = intResult;

    // allocate symmetric memory with the correct size accoring to histogram return value
    alloc.symmMem = (char *) nvshmem_malloc(maxTuplesPerNode * tupleSize);

    // TODO: launch new kernel with the histogram (still in device mem) and the allocated symmetric memory
    // That kernel then should have everything at hand to do the data shuffling using nvshmem put and non-overlapping offsets

}