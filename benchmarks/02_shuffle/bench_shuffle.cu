#include <iostream>

#define DISABLE_ALL_SHUFFLE_PRINTS
#include "shuffle_data.tpp"
#include "shuffle.tpp"

#include "nvshmem.h"

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
    const ShuffleResult<Tuple> result = shuffle<OffsetMode::ATOMIC_INCREMENT, SendBufferMode::USE_BUFFER>(
            80, 128, 3,
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


struct BenchmarkArgs {
    uint32_t grid_dim_min;
    uint32_t grid_dim_step;
    uint32_t grid_dim_max;

    uint32_t block_dim_min;
    uint32_t block_dim_step;
    uint32_t block_dim_max;

    uint32_t send_buffer_size_multiplier_min;
    uint32_t send_buffer_size_multiplier_step;
    uint32_t send_buffer_size_multiplier_max;

    uint64_t tuple_count;

    uint32_t shared_mem;
};

template<OffsetMode offset_mode, SendBufferMode send_buffer_mode, uint16_t tuple_size>
void benchmark(BenchmarkArgs args) {
    typedef Tuple<uint64_t, uint8_t[tuple_size - sizeof(uint64_t)]> TupleType;

    assert(tuple_size == sizeof(TupleType));
    int pe = nvshmem_my_pe();

    for(uint32_t grid_dim_tmp = args.grid_dim_min; grid_dim_tmp <= args.grid_dim_max; grid_dim_tmp += args.grid_dim_step) {
        const auto grid_dim = std::max<uint32_t>(grid_dim_tmp, 1);
        for(uint32_t block_dim_tmp = args.block_dim_min; block_dim_tmp <= args.block_dim_max; block_dim_tmp += args.block_dim_step) {
            const auto block_dim = std::max<uint32_t>(block_dim_tmp, 1);
            for(uint32_t send_buffer_size_multiplier_tmp = args.send_buffer_size_multiplier_min;
                send_buffer_size_multiplier_tmp <= args.send_buffer_size_multiplier_max;
                send_buffer_size_multiplier_tmp += args.send_buffer_size_multiplier_step)
            {
                const auto send_buffer_size_multiplier = std::max<uint32_t>(send_buffer_size_multiplier_tmp, 1);

                cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE));

                cudaStream_t stream;
                cudaStreamCreate(&stream);

                TupleType *tuples;
                CUDA_CHECK(cudaMalloc(&tuples, args.tuple_count * sizeof(TupleType)));
                generate_tuples<TupleType><<<80, 1024, 0, stream>>>(tuples, args.tuple_count, pe + 1, 1);
                cudaStreamSynchronize(stream);

                // shuffle data
                const ShuffleResult<TupleType> result = shuffle<offset_mode, send_buffer_mode>(
                        grid_dim, block_dim, send_buffer_size_multiplier,
                        tuples, args.tuple_count,
                        stream, NVSHMEM_TEAM_WORLD);
                free(result.tuples);

                std::cout << "02_shuffle" << ","
                          << args.tuple_count << ","
                          << tuple_size << ","
                          << grid_dim << ","
                          << block_dim << ","
                          << send_buffer_size_multiplier << ","
                          << static_cast<int>(offset_mode) << ","
                          << static_cast<int>(send_buffer_mode) << ","
                          << std::chrono::duration_cast<std::chrono::nanoseconds>(result.shuffle_time + result.histogram_time).count() << ","
                          << gb_per_sec(result.shuffle_time + result.histogram_time, tuple_size * args.tuple_count) << std::endl;

                cudaFree(tuples);
                cudaStreamDestroy(stream);
            }
        }
    }
}

/**
 * Arguments:
 * 0. program path (implicit)
 * 1. grid_dim_min
 * 2. grid_dim_step
 * 3. grid_dim_iterations
 * 4. block_dim_min
 * 5. block_dim_step
 * 6. block_dim_iterations
 * 7. send_buffer_size_multiplier_min
 * 8. send_buffer_size_multiplier_step
 * 9. send_buffer_size_multiplier_iterations
 * 10. tuple_count
 */
int main(int argc, char *argv[]) {
    if (argc != 11) {
        std::cerr << "Usage: "
                  << argv[0]
                  << " <grid_dim_min> <grid_dim_step> <grid_dim_iterations>"
                  << " <block_dim_min> <block_dim_step> <block_dim_iterations>"
                  << " <send_buffer_size_multiplier_min> <send_buffer_size_multiplier_step> <send_buffer_size_multiplier_iterations>"
                  << " <tuple_count>"
                  << std::endl;
        return 1;
    }

    const uint32_t grid_dim_min = std::stoul(argv[1]);
    const uint32_t grid_dim_step = std::stoul(argv[2]);
    const uint32_t grid_dim_iterations = std::stoul(argv[3]);
    assert(grid_dim_step > 0);
    assert(grid_dim_iterations > 0);

    const uint32_t block_dim_min = std::stoul(argv[4]);
    const uint32_t block_dim_step = std::stoul(argv[5]);
    const uint32_t block_dim_iterations = std::stoul(argv[6]);
    assert(block_dim_step > 0);
    assert(block_dim_iterations > 0);

    const uint32_t send_buffer_size_multiplier_min = std::stoul(argv[7]);
    const uint32_t send_buffer_size_multiplier_step = std::stoul(argv[8]);
    const uint32_t send_buffer_size_multiplier_iterations = std::stoul(argv[9]);
    assert(send_buffer_size_multiplier_step > 0);
    assert(send_buffer_size_multiplier_iterations > 0);

    const uint64_t tuple_count = std::stoull(argv[10]);

    const uint32_t grid_dim_max = grid_dim_min + grid_dim_step * (grid_dim_iterations - 1);
    const uint32_t block_dim_max = block_dim_min + block_dim_step * (block_dim_iterations - 1);
    const uint32_t send_buffer_size_multiplier_max = send_buffer_size_multiplier_min + send_buffer_size_multiplier_step * (send_buffer_size_multiplier_iterations - 1);

    assert(block_dim_max <= 1024);
    uint32_t shared_mem = 4096;

    nvshmem_init();

    std::cout << "type,tuple_count,tuple_size,grid_dim,block_dim,send_buffer_size_multiplier,offset_mode,send_buffer_mode,scan_time_nanoseconds,throughput_gb_s" << std::endl;
    BenchmarkArgs benchmarkArgs(grid_dim_min, grid_dim_step, grid_dim_max,
                                block_dim_min, block_dim_step, block_dim_max,
                                send_buffer_size_multiplier_min, send_buffer_size_multiplier_step, send_buffer_size_multiplier_max,
                                tuple_count, shared_mem);

    benchmark<OffsetMode::SYNC_FREE, SendBufferMode::USE_BUFFER, 32>(benchmarkArgs);
    benchmark<OffsetMode::SYNC_FREE, SendBufferMode::USE_BUFFER, 64>(benchmarkArgs);
    benchmark<OffsetMode::SYNC_FREE, SendBufferMode::USE_BUFFER, 128>(benchmarkArgs);
    benchmark<OffsetMode::SYNC_FREE, SendBufferMode::USE_BUFFER, 256>(benchmarkArgs);
    benchmark<OffsetMode::SYNC_FREE, SendBufferMode::USE_BUFFER, 512>(benchmarkArgs);
    benchmark<OffsetMode::SYNC_FREE, SendBufferMode::USE_BUFFER, 1024>(benchmarkArgs);

    benchmark<OffsetMode::ATOMIC_INCREMENT, SendBufferMode::USE_BUFFER, 32>(benchmarkArgs);
    benchmark<OffsetMode::ATOMIC_INCREMENT, SendBufferMode::USE_BUFFER, 64>(benchmarkArgs);
    benchmark<OffsetMode::ATOMIC_INCREMENT, SendBufferMode::USE_BUFFER, 128>(benchmarkArgs);
    benchmark<OffsetMode::ATOMIC_INCREMENT, SendBufferMode::USE_BUFFER, 256>(benchmarkArgs);
    benchmark<OffsetMode::ATOMIC_INCREMENT, SendBufferMode::USE_BUFFER, 512>(benchmarkArgs);
    benchmark<OffsetMode::ATOMIC_INCREMENT, SendBufferMode::USE_BUFFER, 1024>(benchmarkArgs);

    //benchmark<OffsetMode::SYNC_FREE, SendBufferMode::NO_BUFFER, 32>(benchmarkArgs);
    //benchmark<OffsetMode::SYNC_FREE, SendBufferMode::NO_BUFFER, 64>(benchmarkArgs);
    //benchmark<OffsetMode::SYNC_FREE, SendBufferMode::NO_BUFFER, 128>(benchmarkArgs);
    //benchmark<OffsetMode::SYNC_FREE, SendBufferMode::NO_BUFFER, 256>(benchmarkArgs);
    //benchmark<OffsetMode::SYNC_FREE, SendBufferMode::NO_BUFFER, 512>(benchmarkArgs);
    //benchmark<OffsetMode::SYNC_FREE, SendBufferMode::NO_BUFFER, 1024>(benchmarkArgs);

    //benchmark<OffsetMode::ATOMIC_INCREMENT, SendBufferMode::NO_BUFFER, 32>(benchmarkArgs);
    //benchmark<OffsetMode::ATOMIC_INCREMENT, SendBufferMode::NO_BUFFER, 64>(benchmarkArgs);
    //benchmark<OffsetMode::ATOMIC_INCREMENT, SendBufferMode::NO_BUFFER, 128>(benchmarkArgs);
    //benchmark<OffsetMode::ATOMIC_INCREMENT, SendBufferMode::NO_BUFFER, 256>(benchmarkArgs);
    //benchmark<OffsetMode::ATOMIC_INCREMENT, SendBufferMode::NO_BUFFER, 512>(benchmarkArgs);
    //benchmark<OffsetMode::ATOMIC_INCREMENT, SendBufferMode::NO_BUFFER, 1024>(benchmarkArgs);

    nvshmem_finalize();
    return 0;
}
