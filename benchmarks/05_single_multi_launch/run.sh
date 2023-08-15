#!/bin/bash

input_size=(1000 10000 100000 1000000)

output_file="results.csv"
rm -f $output_file
touch $output_file
echo "type,num_bytes,launches,time_nvshmem,time_rdma" > $output_file

for size in "${input_size[@]}"; do
  echo "Running for input size $size"
  nvshmrun -np 2 ./bench_05_single_multi_launch $size 172.18.94.10 172.18.94.11 > $output_file
done