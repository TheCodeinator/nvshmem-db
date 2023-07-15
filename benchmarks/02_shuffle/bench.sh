#!/bin/bash

for n in 1 2
do
    for ppn in 1 2
    do
        echo "Running with n=$n, ppn=$ppn"
        nvshmrun -n $n -ppn $ppn --hosts 10.0.2.11,10.0.2.12 ./bench_02_shuffle "${n}_${ppn}"
    done
done
