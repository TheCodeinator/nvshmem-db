# Benchmark 03 - packets size

This benchmark tests the time it takes to asynchronously send and receive data of nvshmem_put calls when using different
data sizes in the library call. An expected result would be that we cannot reach network BW with small packet sizes
because then the GPU does not produce enough data to satisfy the bandwidth. With larger msg sizes we expect that we 
can approach the network bandwidth, since the NIC then sould continually be able to send data out.
The benchmark sends the same data repeatedly in order to not have to allocate a large amount of memory.