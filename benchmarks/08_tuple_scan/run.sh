#!/bin/bash

output_file="results.csv"
rm -f $output_file
touch $output_file

nvshmrun -np 1 ./bench_08_tuple_scan 0 8 15 32 32 32 5 5 2 64 64 4 1000000 > $output_file
