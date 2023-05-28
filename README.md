# nvshmem-db

Tested Environment:
- Cmake version 3.10.2 from /usr/bin/cmake on c01
- Compiler: /usr/bin/gcc-10 and /usr/bin/g++-10 on c01
- cuda compiler version: /usr/local/cuda-12/bin/nvcc on c01
  - set cmake variable:  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc
- nvshmem library version 2.9.0 located at /opt/nvshmem/lib/ on c01