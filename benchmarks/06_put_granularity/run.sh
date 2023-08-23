#!/bin/bash

# disable communication over NVLINK or PCI
export NVSHMEM_DISABLE_P2P=true

# 8 GiB
num_bytes=8589934592
# 1 MiB
max_send_si6ze=134217728
# 8 KiB
min_send_size=8192

grids_list=(1 8 64)
blocks_list=(1 32 128)

output_file="results.csv"
rm -f $output_file
touch $output_file
echo "type,num_bytes,max_send_size,message_size,grid_dim,block_dim,throughput" > $output_file

for grid in "${grids_list[@]}"; do
  for block in "${blocks_list[@]}"; do
    echo "Running with 1 host (2 PEs) num_elements=$num_elements, num_blocks=$block"
    nvshmrun -np 2 ./bench_06_put_granularity $grid $block 1 $num_bytes $max_send_size $min_send_size>>$output_file
  done
done

for grid in "${grids_list[@]}"; do
  for block in "${blocks_list[@]}"; do
    echo "Running with 2 hosts (2 PEs) num_elements=$num_elements, num_blocks=$block"
    nvshmrun -n 2 -ppn 1 --hosts 10.0.2.11,10.0.2.12 ./bench_06_put_granularity $grid $block 2 $num_bytes $max_send_size $min_send_size>>$output_file
  done
done
