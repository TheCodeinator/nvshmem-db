#!/bin/bash



# 8 MiB
max_send_size=8388608
# 1 B
min_send_size=1

count=16

grids_list=(1 2 4 8)
blocks_list=(1 8 32 64)

output_file="results.csv"
rm -f $output_file
touch $output_file
echo "type,grid_dim,block_dim,num_hosts,count,max_message_size,message_size,throughput" > $output_file

for grid in "${grids_list[@]}"; do
  for block in "${blocks_list[@]}"; do
    echo "Running with 1 host (2 PEs) NVLINK num_grids=$grid, num_blocks=$block"
    nvshmrun -np 2 ./bench_06_put_granularity $grid $block 0 $count $max_send_size $min_send_size>>$output_file
  done
done

# disable communication over NVLINK or PCI
export NVSHMEM_DISABLE_P2P=true

for grid in "${grids_list[@]}"; do
  for block in "${blocks_list[@]}"; do
    echo "Running with 1 host (2 PEs) NO NVLINK num_grids=$grid, num_blocks=$block"
    nvshmrun -np 2 ./bench_06_put_granularity $grid $block 1 $count $max_send_size $min_send_size>>$output_file
  done
done

for grid in "${grids_list[@]}"; do
  for block in "${blocks_list[@]}"; do
    echo "Running with 2 hosts (2 PEs) num_grids=$grid, num_blocks=$block"
    nvshmrun -n 2 -ppn 1 --hosts 10.0.2.11,10.0.2.12 ./bench_06_put_granularity $grid $block 2 $count $max_send_size $min_send_size>>$output_file
  done
done
