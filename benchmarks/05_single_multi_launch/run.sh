#!/bin/bash

# disable communication over NVLINK or PCI
export PATH=$PATH:/opt/hydra/bin
export NVSHMEM_DISABLE_P2P=true

#input_size=(1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304)
input_size=(1048576,2097152,4194304,8388608,16777216,33554432,67108864,134217728)

output_file="results.csv"
rm -f $output_file
touch $output_file
echo "type,num_bytes,num_bytes_buffer,launches,time_nvshmem,time_rdma" > $output_file

for size in "${input_size[@]}"; do
  echo "Running for input size $size"
  # for each node x ip for ib y is 172.18.94.xy
  nvshmrun -n 2 -ppn 1 --hosts 10.0.2.11,10.0.2.12 ./bench_05_single_multi_launch "$size" 172.18.94.10 172.18.94.20 > $output_file
done