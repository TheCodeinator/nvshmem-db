#include <iostream>
#include "Shuffle.h"

struct shuffle_tuple {
    uint64_t id;
    uint64_t data[7];
};

// configuration for this shuffle example
constexpr uint64_t N_LOCAL_TUPLES = 10;
constexpr uint8_t KEY_OFFSET = 0; // key is first item in shuffle_tuple

// creates local tuples in device memory
shuffle_tuple *createLocalTuples(uint32_t nPes, uint32_t thisPe) {
    constexpr uint64_t localMemSize = N_LOCAL_TUPLES * sizeof(shuffle_tuple);
    // allocate memory for tuples on host
    auto *const localTuplesCPU = static_cast<shuffle_tuple *>(malloc(localMemSize));

    // fill in ids of the tuples as ascending integers with an offset depending on the PE_id
    const uint64_t offset = thisPe * N_LOCAL_TUPLES;
    for (uint64_t i{0}; i < N_LOCAL_TUPLES; ++i) {
        localTuplesCPU[i].id = offset + i;
    }

    // allocate device memory for the local tuples
    shuffle_tuple *localTuplesGPU;
    CUDA_CHECK(cudaMalloc(&localTuplesGPU, N_LOCAL_TUPLES * sizeof(shuffle_tuple)));

    // copy tuples to device memory
    CUDA_CHECK(cudaMemcpy(localTuplesGPU, localTuplesCPU, localMemSize, cudaMemcpyHostToDevice));

    // print local tuples

    for (uint64_t i{0}; i < N_LOCAL_TUPLES; ++i) {
        std::cout << "PE " << thisPe << " has shuffle_tuple " << i << " with key " << localTuplesCPU[i].id << std::endl;
    }

    // free CPU memory
    free(localTuplesCPU);

    return localTuplesGPU;
}

void callShuffle(cudaStream_t &stream, uint64_t nPes, uint64_t thisPe) {
    auto *const localData{createLocalTuples(nPes, thisPe)};

    // shuffle data
    const ShuffleResult result =
        shuffle(reinterpret_cast<uint8_t *>(localData), sizeof(shuffle_tuple), N_LOCAL_TUPLES, KEY_OFFSET, stream, NVSHMEM_TEAM_WORLD);

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
