#!/bin/bash

output_file="results.csv"
rm -f $output_file
touch $output_file

nvshmrun -np 2 ./bench_02_shuffle 8 8 10 128 128 8 0 10 2 5000000 > $output_file
