#!/bin/bash

launches=(2,4,8,16,32,64,128)

output_file="results.csv"
rm -f $output_file
touch $output_file
echo "type,num_bytes,num_bytes_buffer,launches,time_nvshmem,time_rdma" > $output_file

for l in "${launches[@]}"; do
  echo "Running for $l launches"
  ./bench_05_1_single_multi_launch_simple "$l" > $output_file
done