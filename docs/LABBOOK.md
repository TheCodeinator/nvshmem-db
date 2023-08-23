# Project History

This is a living document for issues during the course of the project, document time of the issue and solution.

### 07.06.2023 MPI not installed, not found in cmake 

=> is installed, but in nonstandard location /usr/hpcx/ompi, set -DMPI_HOME=/usr/hpcx/ompi

### 07.06.2023 A couple of paths must be set in bash to find libraries correctly

=> Append the contents of the file ROOT/docs/dev_env.sh to your ~/.bashrc file 

### 06.07.2023 NVSHMEM crash caused by wrong world Team variable
=> cuda crashes because NVSHMEMX_TEAM_NODE must be used instad of NVSHMEM_TEAM_WORLD

### 07.07.2023 Our approach does not realistically capture the problem domain and adapting it would induce further complexity

=> Multiple kernel launches and send operations for data exchange with RDMA are only necessary if the data to send does not fit in the send buffer or the remote does not have enough capacity to store received data. This is the motivation for our approach. However, to capture this realistically, we would have to flush the symmetric upon receipt. This would however need further synchronization to indicate the readiness of the remote to receive again. => Discuss this limitation in our evaluation in the report.
However, a small advantage of our approach with batched send operations is that we can interleave sending with computing. If we computed everything at the beginning and send everything at the end, the send time could not be shadowed with compute time.

### 10.08.2023 Disable P2P transport to force communication over NIC with setting NVSHMEM_DISABLE_P2P=True

=> node local PEs use GPU P2P communication instead of communicating over the NICs. This impedes our ability to soundly 
benchmark RDMA communication (for example with 2 nodes and 2 GPUs per node)