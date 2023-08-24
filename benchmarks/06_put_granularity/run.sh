#!/bin/bash

max_send_size=8388608 # 8 MiB
min_send_size=1

separation_list=(0 1 512)
count=32

grids_list=(1 2 4)
blocks_list=(1 8 32 64)

output_file="results.csv"
rm -f $output_file
touch $output_file
echo "type,grid_dim,block_dim,num_hosts,data_separation,count,max_message_size,message_size,num_bytes,throughput" > $output_file


for grid in "${grids_list[@]}"; do
  for block in "${blocks_list[@]}"; do
    for separation in "${separation_list[@]}"; do
      echo "Running with 1 host (2 PEs) NVLINK num_grids=$grid, num_blocks=$block"
      nvshmrun -np 2 ./bench_06_put_granularity $grid $block 0 $separation  $count $max_send_size $min_send_size>>$output_file
    done
  done
done

# disable communication over NVLINK or PCI
export NVSHMEM_DISABLE_P2P=true

for grid in "${grids_list[@]}"; do
  for block in "${blocks_list[@]}"; do
    for separation in "${separation_list[@]}"; do
      echo "Running with 1 host (2 PEs) NO NVLINK num_grids=$grid, num_blocks=$block"
      nvshmrun -np 2 ./bench_06_put_granularity $grid $block 1 $separation $count $max_send_size $min_send_size>>$output_file
    done
  done
done

for grid in "${grids_list[@]}"; do
  for block in "${blocks_list[@]}"; do
    for separation in "${separation_list[@]}"; do
      echo "Running with 2 hosts (2 PEs) num_grids=$grid, num_blocks=$block"
      nvshmrun -n 2 -ppn 1 --hosts 10.0.2.11,10.0.2.12 ./bench_06_put_granularity $grid $block 2 $separation $count $max_send_size $min_send_size>>$output_file
    done
  done
done
