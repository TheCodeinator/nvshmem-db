#!/bin/bash

nvshmrun -n 2 -ppn 1 --hosts 10.0.2.11,10.0.2.12 ./bench_02_shuffle
