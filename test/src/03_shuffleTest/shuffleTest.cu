#include <iostream>
#include "Shuffle.h"

#define cudaCheckErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"Cuda err detected: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// configuration for this shuffle example
constexpr uint64_t N_LOCAL_TUPLES = 10;
constexpr uint16_t TUPLE_SIZE = 64;
constexpr uint8_t KEY_OFFSET = 0; // key is first item in tuple

// creates local tuples in device memory
char *createLocalTuples(int nPes, int thisPe) {
    constexpr uint64_t localMemSize = N_LOCAL_TUPLES * TUPLE_SIZE;
    // allocate memory for tuples on host
    char *const localTuplesCPU = static_cast<char *>(malloc(localMemSize));

    // fill in ids of the tuples as ascending integers with an offset depending on the PE_id
    const int offset = thisPe * N_LOCAL_TUPLES;
    for (uint64_t i{0}; i < N_LOCAL_TUPLES; ++i) {
        reinterpret_cast<int *>(localTuplesCPU)[i * TUPLE_SIZE] = offset + i;
    }

    // allocate device memory for the local tuples
    char *localTuplesGPU;
    cudaCheckErr(cudaMalloc(&localTuplesGPU, N_LOCAL_TUPLES * TUPLE_SIZE));

    // copy tuples to device memory
    cudaCheckErr(cudaMemcpy(localTuplesGPU, localTuplesCPU, localMemSize, cudaMemcpyHostToDevice));

    // free CPU memory
    free(localTuplesCPU);

    return localTuplesGPU;
}

void callShuffle(cudaStream_t &stream, int nPes, int thisPe) {
    char *const localData{createLocalTuples(nPes, thisPe)};

    // shuffle data
    const ShuffleResult result =
            shuffle(localData, TUPLE_SIZE, N_LOCAL_TUPLES, KEY_OFFSET, stream, NVSHMEM_TEAM_WORLD);

    std::cout << "PE " << thisPe << " has a partition of size " << result.partitionSize << " after shuffling."
              << std::endl;

    // check that correct tuples have been received
    for (uint64_t i{0}; i < result.partitionSize; ++i) {
        // modulo of received tuples should be this PE's ID
        assert(reinterpret_cast<int *>(result.tuples)[i * TUPLE_SIZE] % nPes == thisPe);
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
