# NVSHMEM PUT COALESCING

Test the performance difference between multiple nvshmem put calls and an equivalent nvshmem put call transferring the data in only one call.
We want to investigate whether NVSHMEM buffers the write operations and coalesces them such that multiple subsequent 
contiguous write operations to the same target PE might be similarly performant as one put operation that transfers the
same data in one go.

## CSV Layout

type,node_count,in n,out n
put_coalescing,node_count,in_num_elements,in_num_iteration,in_num_grids,in_num_blocks,out_throughput_one_thread_once,out_throughput_one_thread_sep,out_throughput_multi_thread_sep

