#ifndef NVSHMEM_DB_SHUFFLE_H
#define NVSHMEM_DB_SHUFFLE_H

#include <cuda.h>
#include "nvshmem.h"

#include "shuffle_data.tpp"


template<typename key_type>
__host__ __device__ inline uint32_t distribute(const key_type key, const uint32_t nPes) {
    return key % nPes;
}

enum class OffsetMode {
    ATOMIC_INCREMENT = 0,
    SYNC_FREE = 1
};

enum class SendBufferMode {
    USE_BUFFER = 0,
    NO_BUFFER = 1
};

template<typename Tuple>
struct ShuffleResult {
    std::chrono::nanoseconds histogram_time;
    std::chrono::nanoseconds shuffle_time;
    Tuple *tuples;
    uint64_t partitionSize;
};

template<OffsetMode offset_mode, SendBufferMode send_buffer_mode, typename Tuple>
__host__ ShuffleResult<Tuple> shuffle(
        uint16_t grid_dimension, uint16_t block_dimension, uint8_t send_buffer_size_multiplier,
        const Tuple *device_tuples, uint64_t tuple_count, cudaStream_t const &stream, nvshmem_team_t team
);

#include "shuffle.tpp"

#endif //NVSHMEM_DB_SHUFFLE_H
