#include <iostream>

#include "shuffle_data.tpp"
#include "shuffle.tpp"

#include "nvshmem.h"

template<typename Tuple>
__global__ void printGPUTuples(Tuple *tuples, uint64_t numTuples, int thisPe) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("GPU PE %d start device_tuples: ", thisPe);
        for (uint64_t i = 0; i < numTuples; ++i) {
            printf("%lu ", tuples[i].key);
        }
        printf("\n");
    }
}

// before shuffle::
// PE 0: 0 1 2 3 4
// PE 1: 5 6 7 8 9 10 11 12

// after shuffle:
// PE 0: 0 2 4 6 8 10 12
// PE 1: 1 3 5 7 9 11
template<typename Tuple>
ShuffleResult<Tuple> call_shuffle(cudaStream_t &stream, uint64_t tuple_count) {

    int pe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    int pe_count = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);

    Tuple *tuples;
    CUDA_CHECK(cudaMalloc(&tuples, tuple_count * sizeof(Tuple)));
    generate_tuples<Tuple><<<80, 1024, 0, stream>>>(tuples, tuple_count, pe + 1, 1);
    cudaStreamSynchronize(stream);

#ifndef NDEBUG
    printGPUTuples<<<1, 1, 4096, stream>>>(tuples, tuple_count, pe);
#endif

    // shuffle data
    const ShuffleResult<Tuple> result = shuffle<OffsetMode::SYNC_FREE, SendBufferMode::USE_BUFFER>(
            60, 128, 3,
            tuples, tuple_count,
            stream, NVSHMEM_TEAM_WORLD);

    if (result.partitionSize == 0) {
        throw std::runtime_error("PE " + std::to_string(pe) + " received no tuples");
    }
    for (uint64_t i = 0; i < result.partitionSize; ++i) {
        if (result.tuples[i].key == 0 || distribute(result.tuples[i].key, pe_count) != pe) {
            throw std::runtime_error("PE " + std::to_string(pe) + " received invalid tuple " + std::to_string(result.tuples[i].key) + " at index " + std::to_string(i) + " (partition size: " + std::to_string(result.partitionSize) + ")");
        }
    }

    free(result.tuples);
    return result;
}

int main() {
    constexpr uint64_t tuple_count = 1000000;
    typedef Tuple<uint64_t, uint64_t[31]> TupleType;

    cudaStream_t stream;

    nvshmem_init();
    cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE));
    cudaStreamCreate(&stream);

    call_shuffle<TupleType>(stream, tuple_count);

    nvshmem_finalize();
    return 0;
}
