#!/bin/bash

output_file="results.csv"
rm -f $output_file
touch $output_file

nvshmrun -np 1 ./bench_08_tuple_scan 0 4 21 64 64 16 5 5 1 10000000 > $output_file
