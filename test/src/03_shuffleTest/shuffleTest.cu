#include "SomeLibraryInterfaceFunctions.h"

// TODO: use testing framework such as Boost test, google test or catch2

__global__ void deviceFunction() {
    // call the shuffle function with some nonsense
    shuffle(nullptr, 1, 1, 1, 1, 1, 1);
}

int main() {
    int mype_node, msg;
    cudaStream_t stream;

    nvshmem_init();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream);

    int *destination = (int *) nvshmem_malloc(sizeof(int));

    deviceFunction<<<1, 1, 0, stream>>>();
    nvshmemx_barrier_all_on_stream(stream);
    cudaMemcpyAsync(&msg, destination, sizeof(int), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    printf("%d: received message %d\n", nvshmem_my_pe(), msg);

    nvshmem_free(destination);
    nvshmem_finalize();
    return 0;
}
