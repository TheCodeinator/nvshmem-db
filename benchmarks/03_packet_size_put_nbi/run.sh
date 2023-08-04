#!/bin/bash

grids_list=(1 8 64)
blocks_list=(1 8 64)

output_file="results.csv"
rm -f $output_file
touch $output_file
echo "type,node_count,in_num_bytes,in_num_grids,in_num_blocks,out_throughput" >$output_file
for grid in "${grids_list[@]}"; do
  for block in "${blocks_list[@]}"; do
    echo "Running with 1 host (2 PEs) num_elements=bytes, num_blocks=$block"
    nvshmrun -np 2 ./bench_03_packet_size_put_nbi 4 $grid $block 1 >>$output_file
  done
done

for grid in "${grids_list[@]}"; do
  for block in "${blocks_list[@]}"; do
    echo "Running with 2 hosts (2 PEs) num_elements=$bytes, num_blocks=$block"
    nvshmrun -n 2 -ppn 1 --hosts 10.0.2.11,10.0.2.12 ./bench_03_packet_size_put_nbi 4 $grid $block 2 >>$output_file
  done
done
