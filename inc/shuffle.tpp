#ifndef NVSHMEM_DB_SHUFFLE_TPP
#define NVSHMEM_DB_SHUFFLE_TPP

#include "shuffle.h"


struct ComputeOffsetsResult {
    uint32_t maxPartitionSize;
    uint32_t thisPartitionSize;
};

template <OffsetMode offset_mode, typename Tuple>
__device__ void histLocalAtomic(const uint32_t tid,
                                uint32_t *const hist,
                                const ShuffleData<Tuple> *data,
                                ThreadOffsets<Tuple> *thread_offsets) {
    for (uint32_t i = tid; i < data->tuple_count; i += data->thread_count) {
        // get pointer to the i-th shuffle_tuple
        const Tuple &tuple = data->device_tuples[i];
        // get destination PE ID of the shuffle_tuple
        const uint32_t dest = distribute(tuple.key, data->pe_count);

        // increment corresponding index in compute_offsets
        atomicAdd(hist + dest, 1);
        // increment the count for this thread for the current batch and destination (translate it to the offset later)
        ++(*thread_offsets->getOffset(i / data->send_buffer_size_in_tuples, tid, dest));
    }

    if constexpr(offset_mode == OffsetMode::SYNC_FREE) {
        for (uint32_t i = tid; i < data->batch_count * data->pe_count; i += data->thread_count) {
            uint32_t batch = i / data->pe_count;
            uint32_t dest = i % data->pe_count;
            uint32_t currentOffset = 0;
            for (uint32_t thread = 0; thread < data->thread_count; ++thread) {
                uint32_t *offset = thread_offsets->getOffset(batch, thread, dest);
                uint32_t tmp = *offset;
                *offset = currentOffset;
                currentOffset += tmp;
            }
        }
    }
}

/**
 * Computes a write-offset for each destination PE based on the histograms of each PE.
 * For example, with 4 nodes (zero-indexed), node 3 needs to know the following offsets for its send operations:
  offset(3->0) := 0->0 + 1->0 + 2->0
  offset(3->1) := 0->1 + 1->1 + 2->1
  offset(3->2) := 0->2 + 1->2 + 2->2
  offset(3->3) := 0->3 + 1->3 + 2->3
 */
inline __device__ void offsetsFromHistograms(const uint32_t tid,
                                             const uint32_t thread_count,
                                             const uint32_t pe,
                                             const uint32_t pe_count,
                                             const uint32_t *const histograms,
                                             uint32_t *const offsets) {
    for(uint64_t i = tid; i < pe_count * pe; i += thread_count) {
        uint32_t destPe = i % pe_count;
        uint32_t source = i / pe_count;
        // for each PE get the number of device_tuples for this destination stored in the histogram
        const uint32_t histStart = source * pe_count;
        offsets[destPe] += histograms[histStart + destPe];
    }
}

/**
 * returns the maximum size of a destination partition based on histograms of all PEs,
 * which is the maximum sum of all device_tuples that all PEs send to a destination.
 */
inline __device__ uint32_t maxPartitionSize(uint32_t tid, uint32_t thread_count, const uint32_t *const histograms, const uint32_t pe_count) {
    // TODO: parallelize using GPU threads
    uint32_t max = 0;
    for(uint64_t i = tid; i < pe_count * pe_count; i += thread_count) {
        uint32_t dest = i % pe_count;
        uint32_t source = i / pe_count;
        // for each PE get the number of device_tuples for this destination stored in the histogram
        const uint32_t histStart = source * pe_count;
        uint32_t sumOfTuples = 0;
        for(uint32_t pe{0}; pe < pe_count; ++pe) {
            sumOfTuples += histograms[histStart + dest];
        }
        if(sumOfTuples > max) {
            max = sumOfTuples;
        }
    }

    return max;
}

inline __device__ uint32_t thisPartitionSize(const uint32_t *const histograms, const uint32_t nPes, const uint32_t thisPe) {
    // TODO: parallelize using GPU threads
    uint32_t size = 0;
    for (uint32_t pe{0}; pe < nPes; ++pe) {
        const uint32_t histStart = pe * nPes;
        size += histograms[histStart + thisPe]; // add the num of device_tuples each PE has for this PE
    }
    return size;
}

/**
* Computes a global compute_offsets of the data based on the shuffling destination over all nodes.
* Puts the compute_offsets into the device memory pointer hist.
* Returns the maximum number of device_tuples per node, i.e. the maximum value in the compute_offsets to the host code via
* the device variable intResult.
 * @param localData device memory pointer to data local to this node
 * @param tupleSize size of one shuffle_tuple
 * @param tupleCount number of device_tuples
 * @param keyOffset position where to find the 4-byte integer key
 * @param team team used for nvshmem communication
 * @param nPes number of members in the team
 * @param hist device memory pointer to an int array of size pe_count to store the compute_offsets
 */
template <OffsetMode offset_mode, typename Tuple>
__global__ void compute_offsets(
        const nvshmem_team_t team,
        const int thisPe,
        uint32_t *localHistogram,
        uint32_t *const globalHistograms,
        uint32_t *const offsets,
        ComputeOffsetsResult *offsetsResult,
        const ShuffleData<Tuple> *data,
        ThreadOffsets<Tuple> *thread_offsets) {

    const uint32_t tid = global_thread_id();

    // local histogram will be part of the array. globalHistograms are ordered by PE ID. Each histogram has pe_count elements
    // compute local histogram for this PE
    histLocalAtomic<offset_mode>(tid, localHistogram, data, thread_offsets);

    // TODO: alltoall doesn't work, but fcollect does?
    if(tid == 0) {
        //assert(nvshmem_uint32_fcollect(team, globalHistograms, localHistogram, data->pe_count) == 0); // not working!
        fucking_fcollect(team, globalHistograms, localHistogram, data->pe_count);
    }

#ifndef NDEBUG
    if (tid == 0) {
        printf("PE %d globalHistograms AFTER exchange: %d %d %d %d\n", thisPe, globalHistograms[0], globalHistograms[1],
               globalHistograms[2], globalHistograms[3]);
    }
#endif

    // compute offsets based on globalHistograms
    offsetsFromHistograms(tid, data->thread_count, thisPe, data->pe_count, globalHistograms, offsets);

#ifndef NDEBUG
    // print offsets
    if (tid == 0) {
        printf("(Remote) offsets for PE %d: ", thisPe);
        for (int i{0}; i < data->pe_count; ++i) {
            printf("%d ", offsets[i]);
        }
        printf("\n");
    }
#endif

    // compute max output partition size (max value of histogram sums)
    offsetsResult->maxPartitionSize = maxPartitionSize(tid, data->thread_count, globalHistograms, data->pe_count);

    // compute size of this partition
    offsetsResult->thisPartitionSize = thisPartitionSize(globalHistograms, data->pe_count, thisPe);

#ifndef NDEBUG
    if (tid == 0) {
        printf("PE %d: maxPartitionSize = %d, thisPartitionSize = %d\n", thisPe, offsetsResult->maxPartitionSize,
               offsetsResult->thisPartitionSize);
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
__device__ void async_send_buffers(uint32_t pe, uint32_t thread_id, uint32_t tuple_id, const uint32_t *offsets,
                                   Tuple *const symm_mem, uint32_t *const positions_remote, const ShuffleData<Tuple> *data,
                                   ThreadOffsets<Tuple> *thread_offsets, SendBuffers<Tuple> *send_buffers) {
    // send data out
    for (uint32_t dest = thread_id; dest < data->pe_count; dest += data->thread_count) {
        uint32_t send_tuple_count;
        if constexpr (offset_mode == OffsetMode::SYNC_FREE) {
            send_tuple_count = *thread_offsets->getOffset(tuple_id / data->send_buffer_size_in_tuples, data->thread_count - 1, dest);
        } else if constexpr (offset_mode == OffsetMode::ATOMIC_INCREMENT) {
            send_tuple_count = send_buffers->currentOffsets()[dest];
        }

#ifndef NDEBUG
        printf("PE %d, Thread %d: sends %d device_tuples to PE %d at offset %d\n",
               pe, thread_id, send_tuple_count, dest, offsets[dest] + positions_remote[dest]);
#endif

        // send data to remote PE
        nvshmem_putmem_nbi(reinterpret_cast<void*>(symm_mem + (offsets[dest] + positions_remote[dest])),
                           send_buffers->currentBuffer() + (dest * data->send_buffer_size_in_tuples),
                           send_tuple_count * data->tuple_size,
                           dest);
        // increment the position pointer for remote writing for this PE
        positions_remote[dest] += send_tuple_count;
    }
}

template<OffsetMode offset_mode, SendBufferMode send_buffer_mode, typename Tuple>
__global__ void shuffle_with_offset(const nvshmem_team_t team,
                                    const uint32_t pe,
                                    const uint32_t *const offsets,
                                    Tuple *const symm_mem,
                                    ShuffleData<Tuple> *data,
                                    ThreadOffsets<Tuple> *thread_offsets,
                                    SendBuffers<Tuple> *send_buffers) {
    assert(data->send_buffer_size_in_bytes % data->tuple_size == 0); // buffer size must be a multiple of tuple size
    assert(data->send_buffer_size_in_tuples % data->thread_count == 0); // and it must be a multiple of thread count

    const auto tid = global_thread_id();
    // current position writing to the remote locations
    auto positionsRemote = new uint32_t[data->pe_count];

    const uint iterationToSend = data->send_buffer_size_in_tuples / data->thread_count;
    uint iteration = 0;

#ifndef NDEBUG
    if (tid == 0) {
        printf("PE %d: %d threads, buffer size in tuple %d\n", pe, data->thread_count, data->send_buffer_size_in_tuples);
    }
#endif

    const uint maxIndex = data->thread_count * (data->tuple_count / data->thread_count + (data->tuple_count % data->thread_count != 0));
    // iterate over all local data and compute the destination
    for (uint i = tid; i < maxIndex; i += data->thread_count) {
        uint32_t offset = 0;

        if (i < data->tuple_count) {
            // reference to i-th local tuple
            const Tuple &tuple = data->device_tuples[i];
            // get destination of tuple
            const uint dest = distribute(tuple.key, data->pe_count);

            if constexpr(offset_mode == OffsetMode::SYNC_FREE) {
                auto threadOffset = thread_offsets->getOffset(i / data->send_buffer_size_in_tuples, tid, dest);
                offset = *threadOffset;
                *threadOffset += 1;
            } else if constexpr(offset_mode == OffsetMode::ATOMIC_INCREMENT) {
                // increment the offset for this destination atomically (atomicAdd returns the value before increment)
                offset = atomicAdd(send_buffers->currentOffsets() + dest, 1);
            } else {
                assert(false);
            }

            if constexpr(send_buffer_mode == SendBufferMode::USE_BUFFER) {
                assert(offset < data->send_buffer_size_in_tuples); // assert that offset is not out of bounds
#ifndef NDEBUG
                if(dest == 1) {
                    printf("PE %d, Thread %d: writes tuple id %lu -> %d at offset %d to buffer %d\n", pe, tid,
                           tuple.key, dest, offset,
                           send_buffers->currentBufferIndex());
                }
#endif
                send_buffers->currentBuffer()[dest * data->send_buffer_size_in_tuples + offset] = tuple;
            } else if constexpr(send_buffer_mode == SendBufferMode::NO_BUFFER) {
#ifndef NDEBUG
                printf("PE %d, Thread %d: writes tuple id %lu -> %d at offset %d\n", pe, tid,
                       tuple.key, dest, offsets[dest] + offset);
#endif
                nvshmem_putmem_nbi(reinterpret_cast<void*>(symm_mem + (offsets[dest] + offset)), // position to write in shared mem
                                   &tuple, // tuple to send
                                   data->tuple_size, // number of bytes to send
                                   dest); // destination pe
            } else {
                assert(false);
            }
        }

        // if there might be a full buffer or tuple count reached, send data out
        // We do not track all buffersComp individually, because we can only await all async operations at once anyways
        if constexpr(send_buffer_mode == SendBufferMode::USE_BUFFER) {
            if(++iteration % iterationToSend == 0 || i + (data->thread_count - tid) >= data->tuple_count) {
                if(tid < data->pe_count) {
                    nvshmem_quiet(); // wait for previous send to be completed => buffersBackup reusable after quiet finishes
                }
                __syncthreads(); // sync threads before send operation (to ensure that all threads have written their data into the buffer)
                // send data parallelized and asyncronously to all destinations
                async_send_buffers<offset_mode>(pe, tid, i, offsets, symm_mem, positionsRemote, data, thread_offsets, send_buffers);
                __syncthreads();
                if(tid == 0) {
                    //async_send_buffers<offset_mode>(pe, tid, i, offsets, symm_mem, positionsRemote, data, thread_offsets, send_buffers);
                    send_buffers->useNextBuffer(); // switch to the next buffer
                    send_buffers->resetBuffer(send_buffers->currentBufferIndex()); // reset the offsets of the current buffer
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
    // print only the ID of the tuple
    printf("PE %d result (%lu device_tuples): ", thisPe, tupleCount);
#ifndef NDEBUG
    for (uint64_t i{0}; i < tupleCount; ++i) {
        printf("%lu ", data[i].key);
    }
#endif
    printf("\n");
}


template<OffsetMode offset_mode, SendBufferMode send_buffer_mode, typename Tuple>
__host__ ShuffleResult<Tuple> shuffle(
        uint16_t grid_dimension, uint16_t block_dimension, uint8_t send_buffer_size_multiplier,
        const Tuple *device_tuples, uint64_t tuple_count, cudaStream_t const &stream, nvshmem_team_t team)
{
    const int pe = nvshmem_team_my_pe(team);
    const uint32_t shared_mem = 4 * 1024;

    printf("PE %d: shuffle with tuple_size = %lu, tuple_count = %lu\n", pe, sizeof(Tuple), tuple_count);

    ShuffleData<Tuple> data(device_tuples,
                            nvshmem_team_n_pes(team),
                            grid_dimension,
                            block_dimension,
                            tuple_count,
                            send_buffer_size_multiplier,
                            true);
    SendBuffers<Tuple> send_buffers(&data);
    ThreadOffsets<Tuple> thread_offsets(&data);

    printf("PE %d: shuffle with tuple_size = %d, tuple_count = %lu\n", pe, data.tuple_size, data.tuple_count);

    // allocate symm. memory for the global_histograms of all PEs
    // Each histogram contains pe_count int elements. There is one compute_offsets for each of the pe_count PEs
    auto *global_histograms = static_cast<uint32_t *>(nvshmem_malloc(data.pe_count * data.pe_count * sizeof(uint32_t)));

    // allocate private device memory for storing the write offsets
    // this is sent to all other PEs
    auto *local_histograms = static_cast<uint32_t *>(nvshmem_malloc(data.pe_count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(local_histograms, 0, data.pe_count * sizeof(uint32_t)));

    // allocate private device memory for storing the write offsets;
    // One write offset for each destination PE
    uint32_t *offsets;
    CUDA_CHECK(cudaMalloc(&offsets, data.pe_count * sizeof(uint32_t)));

    // allocate private device memory for the result of the compute_offsets function
    ComputeOffsetsResult *device_offsets_results;
    CUDA_CHECK(cudaMalloc(&device_offsets_results, sizeof(ComputeOffsetsResult)));

    // TODO: What value should the "sharedMem" argument for the collective launch have?
    // compute and exchange the global_histograms and compute the offsets for remote writing
    const auto histogramMeasurement = time_kernel(compute_offsets<offset_mode, Tuple>, grid_dimension, block_dimension, shared_mem, stream,
                                                  team, pe, local_histograms, global_histograms, offsets, device_offsets_results,
                                                  data.device_data, thread_offsets.device_offsets);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // get result from kernel launch
    ComputeOffsetsResult offsets_results{};
    CUDA_CHECK(cudaMemcpy(&offsets_results, device_offsets_results, sizeof(ComputeOffsetsResult), cudaMemcpyDeviceToHost));

    // histograms no longer required since offsets have been calculated => release corresponding symm. memory
    nvshmem_free(global_histograms);
    nvshmem_free(local_histograms);

    // allocate symmetric memory big enough to fit the largest partition
    printf("PE %d: Allocating: %u bytes of symmetric memory for device_tuples after shuffle (%d device_tuples)\n",
           pe, offsets_results.maxPartitionSize * data.tuple_size, offsets_results.maxPartitionSize);
    auto *const symm_mem = reinterpret_cast<Tuple*>(nvshmem_malloc(offsets_results.maxPartitionSize * data.tuple_size));
    CUDA_CHECK(cudaMemset(symm_mem, 0, offsets_results.maxPartitionSize * data.tuple_size));

    printf("Calling shuffleWithOffsets with %d PEs\n", data.pe_count);

    if constexpr(send_buffer_mode == SendBufferMode::NO_BUFFER) {
        nvshmemx_buffer_register(const_cast<void*>(reinterpret_cast<const void*>(data.device_tuples)), data.tuple_count * data.tuple_size);
    }

    // execute the shuffle on the GPU
    //void *shuffleArgs[] = {&team, &thisPe, &offsets,
    //                       const_cast<uint8_t **>(&symm_mem), &deviceShuffleData};
    //NVSHMEM_CHECK(nvshmemx_collective_launch((const void *) shuffle_with_offset<offsetMode, sendBufferMode>, gridDimension, blockDimension, shuffleArgs, 1024 * 4, stream));
    const auto shuffleMeasurement = time_kernel(shuffle_with_offset<offset_mode, send_buffer_mode, Tuple>, grid_dimension, block_dimension, shared_mem, stream,
                                                team, pe, offsets, symm_mem,
                                                data.device_data, thread_offsets.device_offsets, send_buffers.device_buffers);

    CUDA_CHECK(cudaStreamSynchronize(stream)); // wait for kernel to finish and deliver result
    nvshmem_barrier(team); // wait for all send operations to finish

    if constexpr(send_buffer_mode == SendBufferMode::NO_BUFFER) {
        nvshmemx_buffer_unregister(const_cast<void*>(reinterpret_cast<const void*>(data.device_tuples)));
    }

    for(int i = 0; i < data.pe_count; i++) {
        nvshmem_barrier(team);
        if(i == pe) {
            print_tuple_result<<<1, 1, shared_mem, stream>>>(pe, symm_mem, data.tuple_size, offsets_results.thisPartitionSize);
            CUDA_CHECK(cudaStreamSynchronize(stream)); // wait for kernel to finish
        }
        nvshmem_barrier(team);
    }

    printf("PE %d: shuffle GB/s: %f\n", pe, gb_per_sec(shuffleMeasurement, data.tuple_count * data.tuple_size));

    ShuffleResult<Tuple> result{};
    result.partitionSize = offsets_results.thisPartitionSize;
    result.tuples = reinterpret_cast<Tuple*>(malloc(offsets_results.thisPartitionSize * data.tuple_size));
    CUDA_CHECK(cudaMemcpy(result.tuples, symm_mem, offsets_results.thisPartitionSize * data.tuple_size, cudaMemcpyDeviceToHost));
    nvshmem_free(symm_mem);
    return result;
}

#endif //NVSHMEM_DB_SHUFFLE_TPP
