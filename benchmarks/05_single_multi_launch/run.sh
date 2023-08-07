#!/bin/bash

input_size = (1000,10000,100000,1000000)

output_file="results.csv"
rm -f $output_file
touch $output_file
echo "type,num_bytes,launches,time_nvshmem,time_rdma" > $output_file

for size in "${input_size[@]}"; do
  echo "Running for input size $size"
  nvshmrun -n 2 -ppn 1 --hosts 10.0.2.11,10.0.2.12 ./bench_05_single_multi_launch $size 10.0.2.11 10.0.2.12 >> $output_file
done