#include "baseline_shuffle.h"

// used to check the status code of cuda routines for errors
#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t _CHECK_result = (stmt);                                              \
        if (cudaSuccess != _CHECK_result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(_CHECK_result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)


//#################################################
/*
Reusable code that can be identical to the NVSHMEM-based shuffle implementation
TODO: ? Move this to a separate .cu file to make synching changes easier
*/


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

        // increment corresponding index in compute_offsets
        atomicAdd(hist + dest, 1);
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

//#################################
/*
    Code that differs from NVSHMEM-based implementation
*/



/**
    Initiate exchange of the send buffers for each participating node
*/
__host__ void rdma_exchange_data(){
}

ShuffleResult shuffle(
        const uint8_t *const localData, // ptr to device data
        uint16_t tupleSize,
        uint64_t tupleCount,
        uint8_t keyOffset,
        std::string ip,
        uint16_t port,
        std::vector<std::tuple<std::string,uint16_t>> peers,
        const cudaStream_t &stream) {

    // nPes is number of processes participating
    uint32_t n_pes = peers.size();
    size_t globalHistogramsSize = n_pes * n_pes * sizeof(uint32_t);


    //printf("PE %d: shuffle with tupleSize = %d, tupleCount = %lu, keyOffset = %d\n", thisPe, tupleSize, tupleCount,
    //       keyOffset);

    uint32_t * d_global_histograms;
    CUDA_CHECK(cudaMalloc(&d_global_histograms, globalHistogramsSize));

    uint32_t * d_local_histogram;
    CUDA_CHECK(cudaMalloc(&d_local_histogram, n_pes*sizeof(uint32_t)));

    // init rdma

    // call kernels

    // rdma exchange

    // repeat

    // finalize rdma

    ShuffleResult result;

    return result;
}