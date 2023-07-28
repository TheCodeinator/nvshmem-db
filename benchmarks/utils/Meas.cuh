#ifndef NVSHMEM_DB_MEAS_CUH
#define NVSHMEM_DB_MEAS_CUH

constexpr long long SHADER_FREQ_KHZ{1530000};

struct Meas {
    long long start = 0;
    long long stop = 0;

    __host__  [[nodiscard]] inline long long clock_diff() const {
        return stop - start;
    }

    __host__  [[nodiscard]] long double time_diff_ms() const {
        return static_cast<long double>(this->clock_diff()) / SHADER_FREQ_KHZ;
    }

    __host__  [[nodiscard]] long double time_diff_s() const {
        return static_cast<long double>(this->time_diff_ms() / 1000);
    }

    __host__ [[nodiscard]] long double get_throughput(const uint64_t n_bytes) const {
        return n_bytes / this->time_diff_s() / 1000000;
    }

    __host__ [[nodiscard]] std::string to_csv() const {
        // TODO: @Alex If you want, use this to implement printing to csv format
        return {};
    }

    __host__ std::string to_string(const uint64_t n_bytes) const {
        return "start=" + std::to_string(start) +
               " stop=" + std::to_string(stop) +
               " clock_diff=" + std::to_string(this->clock_diff()) +
               " (" + std::to_string(this->time_diff_ms()) +
               "ms) throughput=" + std::to_string(this->get_throughput(n_bytes)) + " GB/s";
    }
};

#endif //NVSHMEM_DB_MEAS_CUH
