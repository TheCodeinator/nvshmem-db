#include <iostream>
#include "Shuffle.h"

struct shuffle_tuple {
    uint64_t id;
    uint64_t data[7];
};

// configuration for this shuffle example
constexpr uint8_t KEY_OFFSET = 0; // key is first item in shuffle_tuple

// creates local tuples in device memory
shuffle_tuple *createLocalTuples(const uint64_t *tupleIds, const size_t numTuples, const size_t offset) {
    size_t localMemSize = numTuples * sizeof(shuffle_tuple);
    // allocate memory for tuples on host
    auto *const localTuplesCPU = static_cast<shuffle_tuple *>(malloc(localMemSize));

    // fill in ids of the tuples as ascending integers with an offset depending on the PE_id
    for (size_t i{0}; i < numTuples; ++i) {
        localTuplesCPU[i].id = tupleIds[i];
    }

    // allocate device memory for the local tuples
    shuffle_tuple *localTuplesGPU;
    CUDA_CHECK(cudaMalloc(&localTuplesGPU, numTuples * sizeof(shuffle_tuple)));

    // copy tuples to device memory
    CUDA_CHECK(cudaMemcpy(localTuplesGPU, localTuplesCPU, localMemSize, cudaMemcpyHostToDevice));

    // free CPU memory
    free(localTuplesCPU);

    return localTuplesGPU;
}

__global__ void printGPUTuples(shuffle_tuple *tuples, uint64_t numTuples, int myPe) {
    if (threadIdx.x == 0) {
        printf("GPU PE %d start tuples: ", myPe);
        for (uint64_t i{0}; i < numTuples; ++i) {
            printf("%lu ", tuples[i].id);
        }
        printf("\n");
    }
}

// before shuffle::
// PE 0: 0 1 2 3
// PE 1: 4 5 6 7 8 9 10

// after shuffle:
// PE 0: 0 2 4 6 8 10
// PE 1: 1 3 5 7 9
void callShuffle(cudaStream_t &stream, uint64_t nPes, uint64_t thisPe) {

    size_t numLocalTuples;
    size_t offsetTupleId;
    int myPe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);

    if (myPe == 0) {
        numLocalTuples = 4;
        offsetTupleId = 0;
    } else {
        numLocalTuples = 7;
        offsetTupleId = 4;
    }

    auto *const tupleIds = static_cast<uint64_t *>(malloc(numLocalTuples * sizeof(uint64_t)));
    for (uint64_t i{0}; i < numLocalTuples; ++i) {
        tupleIds[i] = i + offsetTupleId;
    }

    // create local tuples
    shuffle_tuple *const localData = createLocalTuples(tupleIds, numLocalTuples, offsetTupleId);

    // print tuples on GPU

    printGPUTuples<<<1, 1, 0, stream>>>(localData, numLocalTuples, myPe);

    // shuffle data
    const ShuffleResult result =
        shuffle(reinterpret_cast<uint8_t *>(localData), sizeof(shuffle_tuple), numLocalTuples, KEY_OFFSET, stream, NVSHMEM_TEAM_WORLD);

    std::cout << "PE " << thisPe << " has a partition of size " << result.partitionSize << " after shuffling." << std::endl;

    // print tuples

    for (uint64_t i{0}; i < result.partitionSize; ++i) {
        std::cout << "PE " << thisPe << " has shuffle_tuple " << i << " with key "
                  << reinterpret_cast<int *>(result.tuples)[i * sizeof(shuffle_tuple)] << std::endl;
    }

    // check that correct tuples have been received
    for (uint64_t i{0}; i < result.partitionSize; ++i) {
        // modulo of received tuples should be this PE's ID
        assert(reinterpret_cast<int *>(result.tuples)[i * sizeof(shuffle_tuple)] % nPes == thisPe);
    }
}

int main() {
    int nPes, thisPe;
    cudaStream_t stream;

    nvshmem_init();
    thisPe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    nPes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    cudaSetDevice(thisPe);
    cudaStreamCreate(&stream);

    callShuffle(stream, nPes, thisPe);

    nvshmem_finalize();
    return 0;
}
