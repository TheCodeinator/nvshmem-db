#include <iostream>

#include "shuffle_data.tpp"
#include "shuffle.tpp"

#include "nvshmem.h"

template<typename Tuple>
struct create_tuple_result {
    Tuple **tuples;
    uint64_t *num_tuples;
};

// creates local device_tuples in device memory
template<typename Tuple>
Tuple* create_tuples(uint64_t *tuple_ids, size_t num_tuples) {
    size_t localMemSize = num_tuples * sizeof(Tuple);
    // allocate memory for device_tuples on host
    auto *localTuplesCPU = static_cast<Tuple*>(malloc(localMemSize));

    // fill in ids of the device_tuples as ascending integers with an offset depending on the PE_id
    for (size_t i{0}; i < num_tuples; ++i) {
        localTuplesCPU[i].key = tuple_ids[i];
    }

    // allocate device memory for the local device_tuples
    Tuple *localTuplesGPU;
    CUDA_CHECK(cudaMalloc(&localTuplesGPU, num_tuples * sizeof(Tuple)));

    // copy device_tuples to device memory
    CUDA_CHECK(cudaMemcpy(localTuplesGPU, localTuplesCPU, localMemSize, cudaMemcpyHostToDevice));

    // free CPU memory
    free(localTuplesCPU);

    return localTuplesGPU;
}

template<typename Tuple>
create_tuple_result<Tuple> create_all_local_tuples(const uint64_t tuples_per_pe) {
    int nPes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    auto tuples = reinterpret_cast<Tuple**>(malloc(nPes * sizeof(Tuple*)));
    auto num_tuples = reinterpret_cast<uint64_t*>(malloc(nPes * sizeof(uint64_t)));
    for (int i = 0; i < nPes; ++i) {
        const auto tuple_ids = new uint64_t[tuples_per_pe];
        for (int j = 0; j < tuples_per_pe; ++j) {
            tuple_ids[j] = j + i;
        }
#ifndef NDEBUG
        printf("PE %d has tuple ids: ", i);
        for (int j = 0; j < tuples_per_pe; ++j) {
            printf("%lu ", tuple_ids[j]);
        }
        printf("\n");
#endif
        tuples[i] = create_tuples<Tuple>(tuple_ids, tuples_per_pe);
        num_tuples[i] = tuples_per_pe;
    }
    // print num device_tuples for all pes
    for (int i = 0; i < nPes; ++i) {
        printf("PE %d has %lu device_tuples\n", i, num_tuples[i]);
    }
    return create_tuple_result<Tuple>{
        tuples,
        num_tuples
    };
}

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
void call_shuffle(cudaStream_t &stream, Tuple **local_tuples, uint64_t *num_tuples, const uint64_t tuples_per_pe) {

    int thisPe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    int nPes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);

#ifndef NDEBUG
    printGPUTuples<<<1, 1, 4096, stream>>>(local_tuples[thisPe], num_tuples[thisPe], thisPe);
#endif

    // shuffle data
    const ShuffleResult<Tuple> result = shuffle<OffsetMode::SYNC_FREE, SendBufferMode::USE_BUFFER>(
            42, 128, 100,
            local_tuples[thisPe], num_tuples[thisPe],
            stream, NVSHMEM_TEAM_WORLD);

    assert(result.partitionSize > 0); // assume that every pe receives at least 1 tuple
    for (uint64_t i{0}; i < result.partitionSize; ++i) {
        assert(distribute(result.tuples[i].key, nPes) == thisPe);
    }
}

int main() {
    cudaStream_t stream;

    nvshmem_init();
    cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE));
    cudaStreamCreate(&stream);

    constexpr uint64_t tuples_per_pe = 500000;
    const auto tuple_result = create_all_local_tuples<Tuple<uint64_t, uint64_t[7]>>(tuples_per_pe);
    call_shuffle(stream, tuple_result.tuples, tuple_result.num_tuples, tuples_per_pe);

    nvshmem_finalize();
    return 0;
}
