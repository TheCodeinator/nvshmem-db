#!/bin/bash

elements_list=(100 800 6400 51200 409600 3276800)
grids_list=(1 8 64)
blocks_list=(1 8 64)

output_file="results.csv"
rm -f $output_file
touch $output_file
echo "type,node_count,in_num_elements,in_num_iteration,in_num_grids,in_num_blocks,out_throughput_one_thread_sep,out_throughput_one_thread_once,out_throughput_multi_thread_sep" >$output_file
for elem in "${elements_list[@]}"; do
  for grid in "${grids_list[@]}"; do
    for block in "${blocks_list[@]}"; do
      echo "Running with 1 host (2 PEs) num_elements=$elem, num_blocks=$block"
      echo -n "01_put_coalescing,1,$elem,100,$grid,$block" >>$output_file
      nvshmrun -np 2 ./bench_01_put_coalescing $elem 4 $grid $block >>$output_file
    done
  done
done

for elem in "${elements_list[@]}"; do
  for grid in "${grids_list[@]}"; do
    for block in "${blocks_list[@]}"; do
      echo "Running with 2 hosts (2 PEs) num_elements=$elem, num_blocks=$block"
      echo -n "01_put_coalescing,2,$elem,100,$grid,$block" >>$output_file
      nvshmrun -n 2 -ppn 2 --hosts 10.0.2.11,10.0.2.12 ./bench_01_put_coalescing $elem 4 $grid $block >>$output_file
    done
  done
done
