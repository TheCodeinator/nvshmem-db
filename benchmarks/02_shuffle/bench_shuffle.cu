#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include <ctime>
//#include <Shuffle.h>

int main(int argc, char *argv[]) {

    std::ofstream outfile;
    outfile.open("bench_shuffle_out.csv");
    outfile << "type, node_count,in n,out n\n";

    for (int tableSize{1000}; tableSize <= 100000; tableSize *= 2) {

        outfile << "nvshmem_shuffle, 1, " << tableSize << ", ";

        auto start = std::chrono::steady_clock::now();
        sleep(1e-4 * tableSize);
        auto end = std::chrono::steady_clock::now();

        auto dur = end - start;
        auto time_ms = dur.count() * 1e-6;

        outfile << time_ms << "\n";

    }

    outfile.close();

    return EXIT_SUCCESS;
}
