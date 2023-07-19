#include <iostream>
#include <chrono>
#include <fstream>
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

create_tuple_result create_all_local_tuples(int table_size) {
    int nPes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    shuffle_tuple **tuples = (shuffle_tuple **) malloc(nPes * sizeof(shuffle_tuple *));
    uint64_t *num_tuples = (uint64_t *) malloc(nPes * sizeof(uint64_t));

    for (int i = 0; i < nPes; ++i) {
        const uint64_t count_tuples = table_size / nPes;  // distribute the table_size evenly across all PEs
        const auto tuple_ids = new uint64_t[count_tuples];
        for (uint64_t j = 0; j < count_tuples; ++j) {
            tuple_ids[j] = j + count_tuples * i;  // adjusted to ensure unique ids
        }
        tuples[i] = create_tuples(tuple_ids, count_tuples);
        num_tuples[i] = count_tuples;
    }
    return create_tuple_result{
        tuples,
        num_tuples
    };
}

void call_shuffle(cudaStream_t &stream, shuffle_tuple **local_tuples, uint64_t *num_tuples) {

    int thisPe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    int nPes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);

    // shuffle data
    const ShuffleResult result =
        shuffle(reinterpret_cast<const uint8_t *>(local_tuples[thisPe]), sizeof(shuffle_tuple), num_tuples[thisPe],
                KEY_OFFSET, stream, NVSHMEM_TEAM_WORLD);

    for (uint64_t i{0}; i < result.partitionSize; ++i) {
        // modulo of received tuples should be this PE's ID
        assert(reinterpret_cast<uint64_t *>(result.tuples)[i * 8 + KEY_OFFSET] % nPes == thisPe);
    }
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <nhosts_ppn>" << std::endl;
        return 1;
    }

    int nPes, thisPe;
    cudaStream_t stream;

    nvshmem_init();
    thisPe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    cudaStreamCreate(&stream);

    std::string arg(argv[1]);
    std::string filename = "shuffle_" + arg + ".csv";
    std::ofstream outfile;
    outfile.open(filename.c_str());

    outfile << "table_size, time" << std::endl;

    for(int table_size{16};table_size<=100000000;table_size*=2) {
        // Pass the table size to the tuple creation function
        const create_tuple_result tuple_result = create_all_local_tuples(table_size);
        auto start = std::chrono::steady_clock::now();
        call_shuffle(stream, tuple_result.tuples, tuple_result.num_tuples);
        auto end = std::chrono::steady_clock::now();
        auto dur = end-start;
        auto time_ms = dur.count()*1e-6;

        outfile << table_size << ", " << time_ms << "\n";
    }

    nvshmem_finalize();
    return 0;
}
