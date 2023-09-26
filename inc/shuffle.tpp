#ifndef NVSHMEM_DB_SHUFFLE_TPP
#define NVSHMEM_DB_SHUFFLE_TPP

#include "shuffle.h"


struct ComputeOffsetsResult {
    uint32_t *offsets;
    uint64_t max_partition_size;
    uint64_t this_partition_size;
};

/**
 * Computes a write-offset for each destination PE based on the histograms of each PE.
 * For example, with 4 nodes (zero-indexed), node 3 needs to know the following offsets for its send operations:
  offset(3->0) := 0->0 + 1->0 + 2->0
  offset(3->1) := 0->1 + 1->1 + 2->1
  offset(3->2) := 0->2 + 1->2 + 2->2
  offset(3->3) := 0->3 + 1->3 + 2->3
 */
inline __host__ ComputeOffsetsResult offsetsFromHistograms(const uint32_t pe,
                                                           const uint32_t pe_count,
                                                           const uint32_t *const device_global_histograms) {
    auto offsets = static_cast<uint32_t*>(malloc(pe_count * sizeof(uint32_t)));
    memset(offsets, 0, pe_count * sizeof(uint32_t));
    uint64_t this_partition_size = 0;
    uint64_t max_partition_size = 0;

    auto host_histograms = static_cast<uint32_t*>(malloc(pe_count * pe_count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(host_histograms, device_global_histograms, pe_count * pe_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    for (int dest = 0; dest < pe_count; ++dest) { // for each destination PE
        uint32_t tuple_sum = 0;
        for (uint32_t source = 0; source < pe_count; ++source) {
            const uint32_t histStart = source * pe_count;
            tuple_sum += host_histograms[histStart + dest]; // for each pe sending to the destination
            if(source < pe) { // sum up all histograms of all PEs before this PE
                offsets[dest] += host_histograms[histStart + dest];
            }
        }
        max_partition_size = std::max<uint64_t>(max_partition_size, tuple_sum);
        this_partition_size += host_histograms[(dest * pe_count) + pe]; // add the num of device_tuples each PE has for this PE
    }

    free(host_histograms);

#if !defined(NDEBUG) && !defined(DISABLE_ALL_SHUFFLE_PRINTS)
    printf("PE %d: max_partition_size = %lu, this_partition_size = %lu\n",
           pe, max_partition_size, this_partition_size);

    printf("(Remote) offsets for PE %d: ", pe);
    for (int i = 0; i < pe_count; ++i) {
        printf("%d ", offsets[i]);
    }
    printf("\n");
#endif

    return ComputeOffsetsResult {offsets, max_partition_size, this_partition_size};
}

template <OffsetMode offset_mode, typename Tuple>
__device__ void histLocalAtomic(uint32_t *const hist,
                                const ShuffleData<Tuple> *data,
                                ThreadOffsets<Tuple> *thread_offsets) {
    const uint64_t tuple_offset = blockIdx.x * data->tuple_per_block;
    uint64_t max_i = (tuple_offset + data->tuple_per_block < data->tuple_count) ? data->tuple_per_block : llmax(0, static_cast<int64_t>(data->tuple_count) - tuple_offset);

    for(uint64_t i = threadIdx.x; i < max_i; i += blockDim.x) {
        const Tuple &tuple = data->device_tuples[tuple_offset + i];
        const uint32_t dest = distribute(tuple.key, data->pe_count);

        // increment corresponding index in compute_histograms
        auto before = atomicAdd(hist + dest, 1);
        // increment the count for this thread for the current batch and destination (translate it to the offset later)
        ++(*thread_offsets->getOffset(i / data->send_buffer_size_in_tuples, blockIdx.x, threadIdx.x, dest));
    }
    __syncthreads();

    if constexpr(offset_mode == OffsetMode::SYNC_FREE) {
        for (uint32_t i = threadIdx.x; i < data->batch_count * data->pe_count; i += blockDim.x) {
            uint32_t batch = i / data->pe_count;
            uint32_t dest = i % data->pe_count;
            uint16_t currentOffset = 0;
            for (uint32_t thread = 0; thread < blockDim.x; ++thread) {
                uint16_t *offset = thread_offsets->getOffset(batch, blockIdx.x, thread, dest);
                uint16_t tmp = *offset;
                *offset = currentOffset;
                currentOffset += tmp;
            }
        }
#if !defined(NDEBUG) && !defined(DISABLE_ALL_SHUFFLE_PRINTS)
        //__syncthreads();
        //if(nvshmem_my_pe() == 0 && global_thread_id() == 0) {
        //    for(uint32_t block_id = 0; block_id < gridDim.x; ++block_id) {
        //        for(uint32_t dest_id = 0; dest_id < data->pe_count; ++dest_id) {
        //            for(uint32_t thread_id = 0; thread_id < blockDim.x; ++thread_id) {
        //                for(uint32_t batch_id = 0; batch_id < data->batch_count; ++batch_id) {
        //                    printf("PE %d: Block %d, Batch %d, Thread %d, Dest %d: %d\n",
        //                           nvshmem_my_pe(), block_id, batch_id, thread_id, dest_id,
        //                           *thread_offsets->getOffset(batch_id, block_id, thread_id, dest_id));
        //                }
        //            }
        //        }
        //    }
        //}
#endif
    }
}

/**
* Computes a global compute_histograms of the data based on the shuffling destination over all nodes.
* Puts the compute_histograms into the device memory pointer hist.
* Returns the maximum number of device_tuples per node, i.e. the maximum value in the compute_histograms to the host code via
* the device variable intResult.
 * @param localData device memory pointer to data local to this node
 * @param tupleSize size of one shuffle_tuple
 * @param tupleCount number of device_tuples
 * @param keyOffset position where to find the 4-byte integer key
 * @param team team used for nvshmem communication
 * @param nPes number of members in the team
 * @param hist device memory pointer to an int array of size pe_count to store the compute_histograms
 */
template <OffsetMode offset_mode, typename Tuple>
__global__ void compute_histograms(
        const nvshmem_team_t team,
        const int thisPe,
        uint32_t *localHistogram,
        uint32_t *const globalHistograms,
        const ShuffleData<Tuple> *data,
        ThreadOffsets<Tuple> *thread_offsets) {

    const uint32_t thread_id = global_thread_id();
    const uint32_t thread_count = global_thread_count();

    // local histogram will be part of the array. globalHistograms are ordered by PE ID. Each histogram has pe_count elements
    // compute local histogram for this PE
    histLocalAtomic<offset_mode>(localHistogram, data, thread_offsets);

#if !defined(NDEBUG) && !defined(DISABLE_ALL_SHUFFLE_PRINTS)
    if(thread_id == 0) {
        printf("PE %d localHistogram:", thisPe);
        for(int i = 0; i < data->pe_count; ++i) {
            printf(" %d", localHistogram[i]);
        }
        printf("\n");
    }
#endif

    // TODO: alltoall doesn't work, but fcollect does?
    if(thread_id == 0) {
        //assert(nvshmem_uint32_fcollect(team, globalHistograms, localHistogram, data->pe_count) == 0); // not working!
        fucking_fcollect(team, globalHistograms, localHistogram, data->pe_count);
    }

#if !defined(NDEBUG) && !defined(DISABLE_ALL_SHUFFLE_PRINTS)
    if(thread_id == 0) {
        printf("PE %d globalHistograms AFTER exchange:", thisPe);
        for(int i = 0; i < data->pe_count; ++i) {
            printf(" %d", localHistogram[i]);
        }
        printf("\n");
    }
#endif
}


/**
 * Swaps the buffer pointers and position pointers and clears the main buffers
 * Then asynchronously sends results out based on the backup buffers
 * When this function returns, the main buffer and position can be reused (since they have been swapped)
 * This function completes immediately but the networking has to be awaited with nvshmem_quiet before the next call
 * to this function
 */
template <OffsetMode offset_mode, typename Tuple>
__device__ void async_send_buffers(uint32_t pe, uint32_t i, const uint32_t *offsets,
                                   Tuple *const symm_mem, uint32_t *const positions_remote, const ShuffleData<Tuple> *data,
                                   ThreadOffsets<Tuple> *thread_offsets, SendBuffers<Tuple> *send_buffers) {
    // send data out
    for (uint32_t dest = threadIdx.x; dest < data->pe_count; dest += blockDim.x) {
        uint32_t send_tuple_count;
        if constexpr (offset_mode == OffsetMode::SYNC_FREE) {
            send_tuple_count = *thread_offsets->getOffset(i / data->send_buffer_size_in_tuples, blockIdx.x, blockDim.x - 1, dest);
        } else if constexpr (offset_mode == OffsetMode::ATOMIC_INCREMENT) {
            send_tuple_count = send_buffers->currentOffsets(blockIdx.x)[dest];
        }

        const uint32_t remote_position = atomicAdd(positions_remote + dest, send_tuple_count);

#if !defined(NDEBUG) && !defined(DISABLE_ALL_SHUFFLE_PRINTS)
        if(pe == 0) {
            printf("PE %d, Block %d, Thread %d: sends %d device_tuples to PE %d at offset %d\n",
                   pe, blockIdx.x, threadIdx.x, send_tuple_count, dest, offsets[dest] + remote_position);
        }
#endif

        // send data to remote PE
        //nvshmem_putmem_nbi(reinterpret_cast<void*>(symm_mem + (offsets[dest] + remote_position)),
        //                   send_buffers->currentBuffer(blockIdx.x) + (dest * data->send_buffer_size_in_tuples),
        //                   send_tuple_count * data->tuple_size,
        //                   dest);
    }
}

template<OffsetMode offset_mode, SendBufferMode send_buffer_mode, typename Tuple>
__global__ void shuffle_with_offset(const nvshmem_team_t team,
                                    const uint32_t pe,
                                    const uint32_t *const offsets,
                                    Tuple *const symm_mem,
                                    uint32_t *const remote_positions,
                                    ShuffleData<Tuple> *data,
                                    ThreadOffsets<Tuple> *thread_offsets,
                                    SendBuffers<Tuple> *send_buffers) {
    assert(data->send_buffer_size_in_bytes % data->tuple_size == 0); // buffer size must be a multiple of tuple size
    assert(data->send_buffer_size_in_tuples % data->block_dim == 0); // and it must be a multiple of thread count

    const uint16_t iteration_to_send = data->send_buffer_size_in_tuples / blockDim.x;
    uint iteration = 0;

    const uint64_t tuple_offset = blockIdx.x * data->tuple_per_block;

    for(uint64_t i = threadIdx.x; i < data->tuple_per_block; i += blockDim.x) {
        const uint64_t tuple_index = tuple_offset + i;
        if (tuple_index < data->tuple_count) {
            // reference to i-th local tuple
            const Tuple &tuple = data->device_tuples[tuple_index];
            // get destination of tuple
            const uint dest = distribute(tuple.key, data->pe_count);

            uint32_t offset;
            if constexpr(offset_mode == OffsetMode::SYNC_FREE) {
                auto thread_offset = thread_offsets->getOffset(i / data->send_buffer_size_in_tuples, blockIdx.x, threadIdx.x, dest);
                offset = *thread_offset;
                *thread_offset += 1;
            } else if constexpr(offset_mode == OffsetMode::ATOMIC_INCREMENT) {
                // increment the offset for this destination atomically (atomicAdd returns the value before increment)
                offset = atomicAdd(send_buffers->currentOffsets(blockIdx.x) + dest, 1);
            } else {
                assert(false);
            }

            if constexpr(send_buffer_mode == SendBufferMode::USE_BUFFER) {
                assert(offset < data->send_buffer_size_in_tuples); // assert that offset is not out of bounds

#if !defined(NDEBUG) && !defined(DISABLE_ALL_SHUFFLE_PRINTS)
                if(pe == 0) {
                    printf("PE %d, Block %d, Thread %d: writes tuple id %lu -> %d at offset %d to buffer %d\n",
                           pe, blockIdx.x, threadIdx.x,
                           tuple.key, dest, offset, send_buffers->currentBufferIndex(blockIdx.x));
                }
#endif
                send_buffers->currentBuffer(blockIdx.x)[dest * data->send_buffer_size_in_tuples + offset] = tuple;
            } else if constexpr(send_buffer_mode == SendBufferMode::NO_BUFFER) {

#if !defined(NDEBUG) && !defined(DISABLE_ALL_SHUFFLE_PRINTS)
                printf("PE %d, Block %d, Thread %d: writes tuple id %lu -> %d at offset %d\n",
                       pe, blockIdx.x, threadIdx.x,
                       tuple.key, dest, offsets[dest] + offset);
#endif
                nvshmem_putmem_nbi(reinterpret_cast<void*>(symm_mem + (offsets[dest] + offset)), // position to write in shared mem
                                   &tuple, // tuple to send
                                   sizeof(Tuple), // number of bytes to send
                                   dest); // destination pe
            } else {
                assert(false);
            }
        }

        // if there might be a full buffer or tuple count reached, send data out
        // We do not track all buffersComp individually, because we can only await all async operations at once anyways
        if constexpr(send_buffer_mode == SendBufferMode::USE_BUFFER) {
            if(++iteration % iteration_to_send == 0 || i + blockDim.x >= data->tuple_per_block) {
                if(threadIdx.x < data->pe_count) {
                    nvshmem_quiet(); // wait for previous send to be completed => buffersBackup reusable after quiet finishes
                }
                __syncthreads(); // sync threads before send operation (to ensure that all threads have written their data into the buffer)
                // send data parallelized and asyncronously to all destinations
                async_send_buffers<offset_mode>(pe, i, offsets, symm_mem, remote_positions, data, thread_offsets, send_buffers);
                __syncthreads();
                if(threadIdx.x == 0) {
                    //async_send_buffers<offset_mode>(pe, tid, i, offsets, symm_mem, positionsRemote, data, thread_offsets, send_buffers);
                    send_buffers->useNextBuffer(blockIdx.x); // switch to the next buffer
                    send_buffers->resetCurrentBuffer(blockIdx.x); // reset the offsets of the current buffer
                }
                __syncthreads(); // sync threads after send operation
            }
        } else if constexpr(send_buffer_mode == SendBufferMode::NO_BUFFER) {
        } else {
            assert(false);
        }
    }
}

// TODO: use tree-based sum reduction for computing the histograms and the sums if possible:
//  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//  or: https://www.cs.ucr.edu/~amazl001/teaching/cs147/S21/slides/13-Histogram.pdf
//  or: https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/histogram64/doc/histogram.pdf


template<typename Tuple>
__global__ void print_tuple_result(const uint32_t thisPe, const Tuple *const data, const uint16_t tupleSize,
                                   const uint64_t tupleCount) {
#if !defined(DISABLE_ALL_SHUFFLE_PRINTS)
    // print only the ID of the tuple
    printf("PE %d result (%lu device_tuples): ", thisPe, tupleCount);

#if !defined(NDEBUG)
    for (uint64_t i{0}; i < tupleCount; ++i) {
        printf("%lu ", data[i].key);
    }
#endif

    printf("\n");
#endif
}


template<OffsetMode offset_mode, SendBufferMode send_buffer_mode, typename Tuple>
__host__ ShuffleResult<Tuple> shuffle(
        uint16_t grid_dimension, uint16_t block_dimension, uint8_t send_buffer_size_multiplier,
        const Tuple *device_tuples, uint64_t tuple_count, cudaStream_t const &stream, nvshmem_team_t team)
{
    ShuffleResult<Tuple> result;

    const int pe = nvshmem_team_my_pe(team);
    const uint32_t shared_mem = 4 * 1024;

    ShuffleData<Tuple> data(device_tuples,
                            nvshmem_team_n_pes(team),
                            grid_dimension,
                            block_dimension,
                            tuple_count,
                            send_buffer_size_multiplier,
                            true);
    SendBuffers<Tuple> send_buffers(&data);
    ThreadOffsets<Tuple> thread_offsets(&data);

#if !defined(DISABLE_ALL_SHUFFLE_PRINTS)
    printf("PE %d: shuffle with tuple_size = %d, tuple_count = %lu\n", pe, data.tuple_size, data.tuple_count);
#endif

    // allocate symm. memory for the global_histograms of all PEs
    // Each histogram contains pe_count int elements. There is one compute_histograms for each of the pe_count PEs
    auto *global_histograms = static_cast<uint32_t *>(nvshmem_malloc(data.pe_count * data.pe_count * sizeof(uint32_t)));

    // allocate private device memory for storing the write offsets
    // this is sent to all other PEs
    auto *local_histograms = static_cast<uint32_t *>(nvshmem_malloc(data.pe_count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(local_histograms, 0, data.pe_count * sizeof(uint32_t)));

    // TODO: What value should the "sharedMem" argument for the collective launch have?
    // compute and exchange the global_histograms and compute the offsets for remote writing
    result.histogram_time = time_kernel(compute_histograms<offset_mode, Tuple>, grid_dimension, block_dimension,
                                        shared_mem, stream,
                                        team, pe, local_histograms, global_histograms,
                                        data.device_data, thread_offsets.device_offsets);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_barrier(team); // wait for all PEs to finish compute_histograms

    // compute offsets based on globalHistograms
    ComputeOffsetsResult offset_results = offsetsFromHistograms(pe, data.pe_count, global_histograms);
    uint32_t *device_offsets;
    CUDA_CHECK(cudaMalloc(&device_offsets, data.pe_count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(device_offsets, offset_results.offsets, data.pe_count * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // histograms no longer required since offsets have been calculated => release corresponding symm. memory
    nvshmem_free(global_histograms);
    nvshmem_free(local_histograms);

    // allocate symmetric memory big enough to fit the largest partition
#if !defined(DISABLE_ALL_SHUFFLE_PRINTS)
    printf("PE %d: Allocating: %lu bytes of symmetric memory for device_tuples after shuffle (%lu device_tuples)\n",
           pe, offset_results.max_partition_size * data.tuple_size, offset_results.max_partition_size);
#endif
    auto *const symm_mem = reinterpret_cast<Tuple*>(nvshmem_malloc(offset_results.max_partition_size * data.tuple_size));
    CUDA_CHECK(cudaMemset(symm_mem, 0, offset_results.max_partition_size * data.tuple_size));

#if !defined(DISABLE_ALL_SHUFFLE_PRINTS)
    printf("Calling shuffleWithOffsets with %d PEs\n", data.pe_count);
#endif

    if constexpr(send_buffer_mode == SendBufferMode::NO_BUFFER) {
        nvshmemx_buffer_register(const_cast<void*>(reinterpret_cast<const void*>(data.device_tuples)), data.tuple_count * data.tuple_size);
    }

    // current position writing to the remote locations
    uint32_t *remote_positions;
    CUDA_CHECK(cudaMalloc(&remote_positions, data.pe_count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(remote_positions, 0, data.pe_count * sizeof(uint32_t)));

    // execute the shuffle on the GPU
    //void *shuffleArgs[] = {&team, &thisPe, &offsets,
    //                       const_cast<uint8_t **>(&symm_mem), &deviceShuffleData};
    //NVSHMEM_CHECK(nvshmemx_collective_launch((const void *) shuffle_with_offset<offsetMode, sendBufferMode>, gridDimension, blockDimension, shuffleArgs, 1024 * 4, stream));
    result.shuffle_time = time_kernel(shuffle_with_offset<offset_mode, send_buffer_mode, Tuple>, grid_dimension, block_dimension, shared_mem, stream,
                                      team, pe, device_offsets, symm_mem, remote_positions,
                                      data.device_data, thread_offsets.device_offsets, send_buffers.device_buffers);

    CUDA_CHECK(cudaStreamSynchronize(stream)); // wait for kernel to finish and deliver result
    nvshmem_barrier(team); // wait for all send operations to finish

    CUDA_CHECK(cudaFree(remote_positions));

    if constexpr(send_buffer_mode == SendBufferMode::NO_BUFFER) {
        nvshmemx_buffer_unregister(const_cast<void*>(reinterpret_cast<const void*>(data.device_tuples)));
    }

    for(int i = 0; i < data.pe_count; i++) {
        if(i == pe) {
            print_tuple_result<<<1, 1, shared_mem, stream>>>(pe, symm_mem, data.tuple_size, offset_results.this_partition_size);
            CUDA_CHECK(cudaStreamSynchronize(stream)); // wait for kernel to finish
        }
    }

#if !defined(DISABLE_ALL_SHUFFLE_PRINTS)
    printf("PE %d: shuffle GB/s: %f\n", pe, gb_per_sec(result.shuffle_time, data.tuple_count * data.tuple_size));
#endif

    result.partitionSize = offset_results.this_partition_size;
    result.tuples = reinterpret_cast<Tuple*>(malloc(result.partitionSize * data.tuple_size));
    CUDA_CHECK(cudaMemcpy(result.tuples, symm_mem, offset_results.this_partition_size * data.tuple_size, cudaMemcpyDeviceToHost));
    nvshmem_free(symm_mem);
    return result;
}

#endif //NVSHMEM_DB_SHUFFLE_TPP
