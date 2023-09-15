#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include <cuda.h>
#include <vector>
#include "nvshmem.h"
#include "NVSHMEMUtils.cuh"
#include "shuffle.h"
#include "shuffle_data_tmp.h"

__global__ void build_thread_offsets(const ShuffleData *data, ThreadOffsets *offsets) {
    const uint32_t thread_id = global_thread_id();

    for (uint32_t i = thread_id; i < data->tuple_count; i += data->thread_count) {
        const uint8_t *tuple = data->device_tuples + (i * data->tuple_size);
        const uint32_t dest = distribute(tuple, data->key_offset, data->pe_count);

        // increment the count for this thread for the current batch and destination (translate it to the offset later)
        ++(*offsets->getOffset(i / data->send_buffer_size_in_tuples, thread_id, dest));
    }

    __syncthreads();
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

__global__ void generate_tuples(uint64_t tuple_size, uint64_t tuple_count, uint8_t *tuples) {
    const uint32_t thread_id = global_thread_id();
    const uint32_t thread_count = global_thread_count();

    for(uint64_t i = thread_id; i < tuple_count; i += thread_count) {
        *(tuples + (i * tuple_size)) = i;
    }
}

__global__ void tuple_scan(ShuffleData *data, SendBuffers *buffers, ThreadOffsets *offsets) {
    if(data == nullptr || buffers == nullptr || offsets == nullptr)
        return;

    const uint32_t thread_id = global_thread_id();

    const uint iterationToSend = data->send_buffer_size_in_tuples / data->thread_count;
    uint iteration = 0;

    constexpr int offsetMode = 0;

    const uint maxIndex = data->thread_count * static_cast<uint32_t>(ceil(static_cast<double>(data->tuple_count) / data->thread_count));
    for(uint64_t i = thread_id; i < maxIndex; i += data->thread_count) {
        if (i < data->tuple_count) {
            const uint8_t *tuple = data->device_tuples + (i * data->tuple_size);
            const uint dest = distribute(tuple, data->key_offset, data->pe_count);

            uint32_t offset;
            if constexpr(offsetMode == 0) {
                auto thread_offset = offsets->getOffset(i / data->send_buffer_size_in_tuples, thread_id, dest);
                offset = *thread_offset;
                *thread_offset += 1;
            } else if constexpr(offsetMode == 1) {
                // increment the offset for this destination atomically (atomicAdd returns the value before increment)
                offset = atomicAdd(buffers->currentOffsets() + dest, 1);
            } else {
                assert(false);
            }

            memcpy(buffers->currentBuffer() + dest * data->send_buffer_size_in_bytes +
                   offset * data->tuple_size, // to dest-th buffer with offset position
                   tuple,
                   data->tuple_size);
        }

        if(++iteration % iterationToSend == 0 || i + (data->thread_count - thread_id) >= data->tuple_count) {
            __syncthreads(); // sync threads before send operation (to ensure that all threads have written their data into the buffer)
            if(thread_id == 0) {
                buffers->useNextBuffer(); // switch to the next buffer
                buffers->resetBuffer(buffers->currentBufferIndex()); // reset the offsets of the current buffer
            }
            __syncthreads(); // sync threads after send operation
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
 * 10. tuple_size_min
 * 11. tuple_size_step
 * 12. tuple_size_iterations
 * 13. tuple_count
 */
int main(int argc, char *argv[]) {
    if (argc != 14) {
        std::cerr << "Usage: "
            << argv[0]
            << " <grid_dim_min> <grid_dim_step> <grid_dim_iterations>"
            << " <block_dim_min> <block_dim_step> <block_dim_iterations>"
            << " <send_buffer_size_multiplier_min> <send_buffer_size_multiplier_step> <send_buffer_size_multiplier_iterations>"
            << " <tuple_size_min> <tuple_size_step> <tuple_size_iterations>"
            << " <tuple_count>"
            << std::endl;
        return 1;
    }

    cudaStream_t stream;

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

    const uint32_t tuple_size_min = std::stoul(argv[10]);
    const uint32_t tuple_size_step = std::stoul(argv[11]);
    const uint32_t tuple_size_iterations = std::stoul(argv[12]);
    assert(tuple_size_min % 8 == 0);
    assert(tuple_size_step >= 8);
    assert(tuple_size_step % 8 == 0);
    assert(tuple_size_iterations > 0);

    const uint64_t tuple_count = std::stoull(argv[13]);

    const uint32_t grid_dim_max = grid_dim_min + grid_dim_step * (grid_dim_iterations - 1);
    const uint32_t block_dim_max = block_dim_min + block_dim_step * (block_dim_iterations - 1);
    const uint32_t thread_count_max = grid_dim_max * block_dim_max;
    const uint32_t send_buffer_size_multiplier_max = send_buffer_size_multiplier_min + send_buffer_size_multiplier_step * (send_buffer_size_multiplier_iterations - 1);
    const uint32_t tuple_size_max = tuple_size_min + tuple_size_step * (tuple_size_iterations - 1);

    assert(block_dim_max <= 1024);

    nvshmem_init();
    cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE));
    cudaStreamCreate(&stream);

    //std::cout << "grid_dim_max: " << grid_dim_max
    //    << ", block_dim_max: " << block_dim_max
    //    << ", send_buffer_size_multiplier_max: " << send_buffer_size_multiplier_max
    //    << ", tuple_size_max: " << tuple_size_max << std::endl;

    std::cout << "tuple_count,tuple_size,grid_dim,block_dim,send_buffer_size_multiplier,scan_time_nanoseconds,throughput_gb_s" << std::endl;

    for(uint32_t tuple_size_tmp = tuple_size_min; tuple_size_tmp <= tuple_size_max; tuple_size_tmp += tuple_size_step) {
        const auto tuple_size = std::max<uint32_t>(tuple_size_tmp, 8);
        uint8_t *tuples;
        uint32_t tuples_size = tuple_size * tuple_count;
        CUDA_CHECK(cudaMalloc(&tuples, tuples_size));

        //std::cout << "generating " << tuple_count << " tuples of size " << tuple_size << " with grid_dim = "
        //          << grid_dim_max << " and block_dim = " << block_dim_max << std::endl;
        generate_tuples<<<grid_dim_max, block_dim_max, 0, stream>>>(tuple_size, tuple_count, tuples);
        cudaStreamSynchronize(stream);

        time_kernel(tuple_scan, 1, 1, 4 * 1024, stream, nullptr, nullptr, nullptr);

        for(uint32_t grid_dim_tmp = grid_dim_min; grid_dim_tmp <= grid_dim_max; grid_dim_tmp += grid_dim_step) {
            const auto grid_dim = std::max<uint32_t>(grid_dim_tmp, 1);
            for(uint32_t block_dim_tmp = block_dim_min; block_dim_tmp <= block_dim_max; block_dim_tmp += block_dim_step) {
                const auto block_dim = std::max<uint32_t>(block_dim_tmp, 1);
                for(uint32_t send_buffer_size_multiplier_tmp = send_buffer_size_multiplier_min;
                    send_buffer_size_multiplier_tmp <= send_buffer_size_multiplier_max;
                    send_buffer_size_multiplier_tmp += send_buffer_size_multiplier_step)
                {
                    if(grid_dim >= 88 && block_dim >= 800) // quick (but dirty) fix for the "One or more PEs cannot launch error"
                        continue;

                    const auto send_buffer_size_multiplier = std::max<uint32_t>(send_buffer_size_multiplier_tmp, 1);
                    ShuffleData data(tuples,
                                     nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD),
                                     grid_dim * block_dim, tuple_count,
                                     tuple_size,
                                     0,
                                     send_buffer_size_multiplier,
                                     true);
                    SendBuffers send_buffers(&data);
                    ThreadOffsets thread_offsets(&data);

                    time_kernel(build_thread_offsets, grid_dim, block_dim, 16 * 1024, stream, data.device_data,
                                thread_offsets.device_offsets);

                    const auto time_taken = time_kernel(tuple_scan, grid_dim, block_dim, 32 * 1024, stream,
                                                        data.device_data, send_buffers.device_buffers,
                                                        thread_offsets.device_offsets);
                    const auto time_taken_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(time_taken).count();

                    if(cudaGetLastError() == cudaSuccess) {
                        std::cout << tuple_count << ","
                                  << tuple_size << ","
                                  << grid_dim << ","
                                  << block_dim << ","
                                  << send_buffer_size_multiplier << ","
                                  << time_taken_nanoseconds << ","
                                  << gb_per_sec(time_taken, tuples_size) << std::endl;
                    }
                }
            }
        }

        cudaFree(tuples);
    }

    nvshmem_finalize();
    return 0;
}
