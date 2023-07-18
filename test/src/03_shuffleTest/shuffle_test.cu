#include <iostream>
#include "shuffle.h"

struct shuffle_tuple {
    uint64_t id;
    uint64_t data[7];
};

struct create_tuple_result {
    shuffle_tuple **tuples;
    uint64_t *num_tuples;
};

// configuration for this shuffle example
constexpr uint8_t KEY_OFFSET = 0; // key is first item in shuffle_tuple

// creates local tuples in device memory
shuffle_tuple *create_tuples(uint64_t *tuple_ids, size_t num_tuples) {
    size_t localMemSize = num_tuples * sizeof(shuffle_tuple);
    // allocate memory for tuples on host
    auto *localTuplesCPU = static_cast<shuffle_tuple *>(malloc(localMemSize));

    // fill in ids of the tuples as ascending integers with an offset depending on the PE_id
    for (size_t i{0}; i < num_tuples; ++i) {
        localTuplesCPU[i].id = tuple_ids[i];
    }

    // allocate device memory for the local tuples
    shuffle_tuple *localTuplesGPU;
    CUDA_CHECK(cudaMalloc(&localTuplesGPU, num_tuples * sizeof(shuffle_tuple)));

    // copy tuples to device memory
    CUDA_CHECK(cudaMemcpy(localTuplesGPU, localTuplesCPU, localMemSize, cudaMemcpyHostToDevice));

    // free CPU memory
    free(localTuplesCPU);

    return localTuplesGPU;
}

create_tuple_result create_all_local_tuples() {
    int nPes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    shuffle_tuple **tuples = (shuffle_tuple **) malloc(nPes * sizeof(shuffle_tuple *));
    uint64_t *num_tuples = (uint64_t *) malloc(nPes * sizeof(uint64_t));
    for (int i = 0; i < nPes; ++i) {
        const uint64_t count_tuples = 5 + 3 * i;
        const auto tuple_ids = new uint64_t[count_tuples];
        for (int j = 0; j < count_tuples; ++j) {
            tuple_ids[j] = j + 5 * i;
        }
        printf("PE %d has tuple ids: ", i);
        for (int j = 0; j < count_tuples; ++j) {
            printf("%lu ", tuple_ids[j]);
        }
        printf("\n");
        tuples[i] = create_tuples(tuple_ids, count_tuples);
        num_tuples[i] = count_tuples;
    }
    // print num tuples for all pes
    for (int i = 0; i < nPes; ++i) {
        printf("PE %d has %lu tuples\n", i, num_tuples[i]);
    }
    return create_tuple_result{
            tuples,
            num_tuples
    };
}

__global__ void printGPUTuples(shuffle_tuple *tuples, uint64_t numTuples, int thisPe) {
    if (threadIdx.x == 0) {
        printf("GPU PE %d start tuples: ", thisPe);
        for (uint64_t i{0}; i < numTuples; ++i) {
            printf("%lu ", tuples[i].id);
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
void call_shuffle(cudaStream_t &stream, shuffle_tuple **local_tuples, uint64_t *num_tuples) {

    int thisPe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    int nPes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);

    printGPUTuples<<<1, 1, 0, stream>>>(local_tuples[thisPe], num_tuples[thisPe], thisPe);

    // shuffle data
    const ShuffleResult result =
            shuffle(reinterpret_cast<const uint8_t *>(local_tuples[thisPe]), sizeof(shuffle_tuple), num_tuples[thisPe],
                    KEY_OFFSET, stream, NVSHMEM_TEAM_WORLD);

    // check that the local result contains the correct tuples
    for (uint64_t i{0}; i < result.partitionSize; ++i) {
        // modulo of received tuples should be this PE's ID
        assert(reinterpret_cast<uint64_t *>(result.tuples)[i * 8] % nPes == thisPe);
    }
}

int main() {
    int nPes, thisPe;
    cudaStream_t stream;

    nvshmem_init();
    thisPe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
//    cudaSetDevice(thisPe);
    cudaStreamCreate(&stream);

    const create_tuple_result tuple_result = create_all_local_tuples();
    call_shuffle(stream, tuple_result.tuples, tuple_result.num_tuples);

    nvshmem_finalize();
    return 0;
}
