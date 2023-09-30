#!/bin/bash

launches=(2 4 8 16 32 64 128)

output_file="results.csv"
rm -f $output_file
touch $output_file
echo "type,launches,time_single,time_multi" > $output_file

for l in "${launches[@]}"; do
  echo "Running for $l launches"
  ./bench_05_single_multi_launch_simple "$l" >> $output_file
done