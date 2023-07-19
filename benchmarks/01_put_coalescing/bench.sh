#!/bin/bash

mkdir results
echo "put_coalescing,node_count,in_num_elements,in_num_iteration,in_num_grids,in_num_blocks,out_throughput_one_thread_once,out_throughput_one_thread_sep,out_throughput_multi_thread_sep" > $output_file

elements_list=(100 200 300 400 500)

for n in 1 2
do
    for ppn in 1 2
    do
        output_file="results/output_$n_$ppn.csv"
        for elem in "${elements_list[@]}"
        do
            echo "Running with n=$n, ppn=$ppn, num_elements=$elem"
            nvshmrun -n $n -ppn $ppn --hosts 10.0.2.11,10.0.2.12 ./bench_01_put_coalescing $elem 100 1 1 >> $output_file
        done
    done
done
