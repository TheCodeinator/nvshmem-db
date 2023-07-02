# NVSHMEM PUT COALESCING

Test the performance difference between multiple nvshmem put calls and an equivalent nvshmem put call transferring the data in only one call.
We want to investigate whether NVSHMEM buffers the write operations and coalesces them such that multiple subsequent 
contiguous write operations to the same target PE might be similarly performant as one put operation that transfers the
same data in one go.

## CSV Layout

TODO: describe layout here 
