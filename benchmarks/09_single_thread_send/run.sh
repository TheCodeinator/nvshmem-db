#!/bin/bash

bytes_per_send=(8,16,32,64,128)

output_file="results.csv"
rm -f $output_file
touch $output_file
echo "type,bytes_per_send,bandwidth" > $output_file

for size in "${bytes_per_send[@]}"; do
  echo "Running for input size $size"
  nvshmrun -n 2 -ppn 1 --hosts 10.0.2.11,10.0.2.12 ./bench_09_single_thread_send "$size" > $output_file
done