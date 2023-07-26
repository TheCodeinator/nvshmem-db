#!/bin/bash

mkdir -p results

elements_list=(100 200 300 400 500)

output_file="results/output_1_host.csv"
rm -f $output_file
touch $output_file
echo "type,node_count,in_num_elements,in_num_iteration,in_num_grids,in_num_blocks,out_throughput_one_thread_once,out_throughput_one_thread_sep,out_throughput_multi_thread_sep" >$output_file
for elem in "${elements_list[@]}"; do
  echo "Running with 1 host (2 PEs) num_elements=$elem"
  echo "put_coalescing_1_host," >>$output_file
  nvshmrun -np 2 ./bench_01_put_coalescing $elem 100 1 1 >>$output_file
done

output_file="results/output_2_host.csv"
rm -f $output_file
touch $output_file
echo "type,node_count,in_num_elements,in_num_iteration,in_num_grids,in_num_blocks,out_throughput_one_thread_once,out_throughput_one_thread_sep,out_throughput_multi_thread_sep" >$output_file
for elem in "${elements_list[@]}"; do
  echo "Running with 2 hosts (2 PEs) num_elements=$elem"
  echo "put_coalescing_2_host," >>$output_file
  nvshmrun -n 2 -ppn 2 --hosts 10.0.2.11,10.0.2.12 ./bench_01_put_coalescing $elem 100 1 1 >>$output_file
done
