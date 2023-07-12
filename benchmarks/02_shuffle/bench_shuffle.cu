#include <iostream>
#include <cuda.h>
#include "nvshmem.h"

int main(int argc, char *argv[]) {
    // Check if a table size argument is given
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <table_size>" << std::endl;
        return 1;
    }

    // Convert argument to integer
    int table_size = std::stoi(argv[1]);

    int nPes, thisPe;
    cudaStream_t stream;

    nvshmem_init();
    thisPe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(thisPe);
    cudaStreamCreate(&stream);

    printf("PE %d: table size %d\n", thisPe, table_size);

    nvshmem_finalize();
    return 0;
}
