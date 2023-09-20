#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include <cuda.h>
#include <type_traits>
#include <vector>
#include "nvshmem.h"
#include "NVSHMEMUtils.cuh"
#include "shuffle.h"


template<typename Tuple>
__global__ void build_thread_offsets(const ShuffleData<Tuple> *data, ThreadOffsets<Tuple> *offsets) {
    const uint32_t thread_id = global_thread_id();

    for (uint32_t i = thread_id; i < data->tuple_count; i += data->thread_count) {
        const Tuple &tuple = data->device_tuples[i];
        const uint32_t dest = distribute(tuple.key, data->pe_count);

        // increment the count for this thread for the current batch and destination (translate it to the offset later)
        ++(*offsets->getOffset(i / data->send_buffer_size_in_tuples, thread_id, dest));
    }

    for (uint32_t i = thread_id; i < data->batch_count * data->pe_count; i += data->thread_count) {
        uint32_t batch = i / data->pe_count;
        uint32_t dest = i % data->pe_count;
        uint32_t currentOffset = 0;
        for (uint32_t thread = 0; thread < data->thread_count; ++thread) {
            uint32_t *offset = offsets->getOffset(batch, thread, dest);
            uint32_t tmp = *offset;
            *offset = currentOffset;
            currentOffset += tmp;
        }
    }
}

template<typename Tuple>
__global__ void generate_tuples(uint64_t tuple_size, uint64_t tuple_count, Tuple *tuples) {
    const uint32_t thread_id = global_thread_id();
    const uint32_t thread_count = global_thread_count();

    for(decltype(Tuple::key) i = thread_id; i < tuple_count; i += thread_count) {
        tuples[i].key = i;
    }
}

template<OffsetMode offset_mode, typename Tuple>
__global__ void tuple_scan(ShuffleData<Tuple> *data, SendBuffers<Tuple> *buffers, ThreadOffsets<Tuple> *offsets) {
    if(data == nullptr || buffers == nullptr || offsets == nullptr)
        return;

    const uint32_t thread_id = global_thread_id();

    const uint iteration_to_send = data->send_buffer_size_in_tuples / data->thread_count;
    uint iteration = 0;

    const uint max_index = data->thread_count * static_cast<uint32_t>(ceil(static_cast<double>(data->tuple_count) / data->thread_count));
    for(uint64_t i = thread_id; i < max_index; i += data->thread_count) {
        if (i < data->tuple_count) {
            const Tuple &tuple = data->device_tuples[i];
            const uint dest = distribute(tuple.key, data->pe_count);

            uint32_t offset;
            if constexpr(offset_mode == OffsetMode::SYNC_FREE) {
                auto thread_offset = offsets->getOffset(i / data->send_buffer_size_in_tuples, thread_id, dest);
                offset = *thread_offset;
                *thread_offset += 1;
            } else if constexpr(offset_mode == OffsetMode::ATOMIC_INCREMENT) {
                // increment the offset for this destination atomically (atomicAdd returns the value before increment)
                offset = atomicAdd(buffers->currentOffsets() + dest, 1);
            } else {
                assert(false);
            }

            buffers->currentBuffer()[dest * data->send_buffer_size_in_tuples + offset] = tuple;
            //memcpy(buffers->currentBuffer() + dest * data->send_buffer_size_in_bytes +
            //       offset * data->tuple_size, // to dest-th buffer with offset position
            //       tuple,
            //       data->tuple_size);

        }

        if(++iteration % iteration_to_send == 0 || i + (data->thread_count - thread_id) >= data->tuple_count) {
            __syncthreads(); // sync threads before send operation (to ensure that all threads have written their data into the buffer)
            if(thread_id == 0) {
                buffers->useNextBuffer(); // switch to the next buffer
                buffers->resetBuffer(buffers->currentBufferIndex()); // reset the offsets of the current buffer
            }
            __syncthreads(); // sync threads after send operation
        }
    }
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

template<OffsetMode offset_mode, uint16_t tuple_size>
void benchmark(BenchmarkArgs args) {
    typedef Tuple<uint64_t, uint8_t[tuple_size - sizeof(uint64_t)]> TupleType;

    assert(tuple_size == sizeof(TupleType));

    for(uint32_t grid_dim_tmp = args.grid_dim_min; grid_dim_tmp <= args.grid_dim_max; grid_dim_tmp += args.grid_dim_step) {
        const auto grid_dim = std::max<uint32_t>(grid_dim_tmp, 1);
        for(uint32_t block_dim_tmp = args.block_dim_min; block_dim_tmp <= args.block_dim_max; block_dim_tmp += args.block_dim_step) {
            const auto block_dim = std::max<uint32_t>(block_dim_tmp, 1);
            for(uint32_t send_buffer_size_multiplier_tmp = args.send_buffer_size_multiplier_min;
                send_buffer_size_multiplier_tmp <= args.send_buffer_size_multiplier_max;
                send_buffer_size_multiplier_tmp += args.send_buffer_size_multiplier_step)
            {
                cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE));

                cudaStream_t stream;
                cudaStreamCreate(&stream);

                TupleType *tuples;
                uint64_t tuples_size = tuple_size * args.tuple_count;
                CUDA_CHECK(cudaMalloc(&tuples, tuples_size));

                generate_tuples<<<args.grid_dim_max, args.block_dim_max, args.shared_mem, stream>>>(tuple_size, args.tuple_count, tuples);
                cudaStreamSynchronize(stream);

                time_kernel(tuple_scan<offset_mode, TupleType>, 1, 1, args.shared_mem, stream, nullptr, nullptr, nullptr);

                const auto send_buffer_size_multiplier = std::max<uint32_t>(send_buffer_size_multiplier_tmp, 1);
                ShuffleData data(tuples,
                                 nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD),
                                 grid_dim,
                                 block_dim,
                                 args.tuple_count,
                                 send_buffer_size_multiplier,
                                 true);
                SendBuffers send_buffers(&data);
                ThreadOffsets thread_offsets(&data);

                auto build_thread_offsets_time = std::chrono::nanoseconds(0);
                if constexpr(offset_mode == OffsetMode::SYNC_FREE) {
                    build_thread_offsets_time = time_kernel(
                            build_thread_offsets<TupleType>,
                            grid_dim, block_dim, args.shared_mem, stream, data.device_data,
                            thread_offsets.device_offsets);
                }

                const auto scan_time = time_kernel(
                        tuple_scan<offset_mode, TupleType>,
                        grid_dim, block_dim, args.shared_mem, stream,
                        data.device_data, send_buffers.device_buffers,
                        thread_offsets.device_offsets);

                const auto time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(scan_time + build_thread_offsets_time).count();

                if(cudaGetLastError() == cudaSuccess) {
                    std::cout << "08_tuple_scan" << ","
                              << args.tuple_count << ","
                              << tuple_size << ","
                              << grid_dim << ","
                              << block_dim << ","
                              << send_buffer_size_multiplier << ","
                              << static_cast<int>(offset_mode) << ","
                              << time_taken << ","
                              << gb_per_sec(scan_time + build_thread_offsets_time, tuples_size) << std::endl;
                }

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

    std::cout << "type,tuple_count,tuple_size,grid_dim,block_dim,send_buffer_size_multiplier,offset_mode,scan_time_nanoseconds,throughput_gb_s" << std::endl;
    BenchmarkArgs benchmarkArgs(grid_dim_min, grid_dim_step, grid_dim_max,
                                   block_dim_min, block_dim_step, block_dim_max,
                                   send_buffer_size_multiplier_min, send_buffer_size_multiplier_step, send_buffer_size_multiplier_max,
                                   tuple_count,
                                   shared_mem);

    benchmark<OffsetMode::SYNC_FREE, 32>(benchmarkArgs);
    benchmark<OffsetMode::SYNC_FREE, 64>(benchmarkArgs);
    benchmark<OffsetMode::SYNC_FREE, 128>(benchmarkArgs);
    benchmark<OffsetMode::SYNC_FREE, 256>(benchmarkArgs);
    benchmark<OffsetMode::SYNC_FREE, 512>(benchmarkArgs);
    benchmark<OffsetMode::SYNC_FREE, 1024>(benchmarkArgs);

    benchmark<OffsetMode::ATOMIC_INCREMENT, 32>(benchmarkArgs);
    benchmark<OffsetMode::ATOMIC_INCREMENT, 64>(benchmarkArgs);
    benchmark<OffsetMode::ATOMIC_INCREMENT, 128>(benchmarkArgs);
    benchmark<OffsetMode::ATOMIC_INCREMENT, 256>(benchmarkArgs);
    benchmark<OffsetMode::ATOMIC_INCREMENT, 512>(benchmarkArgs);
    benchmark<OffsetMode::ATOMIC_INCREMENT, 1024>(benchmarkArgs);

    nvshmem_finalize();
    return 0;
}
