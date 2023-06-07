# nvshmem-db

Tested Environment:
- Compiler: /usr/bin/gcc-10 and /usr/bin/g++-10 on c01
- cuda compiler version: /usr/local/cuda-12/bin/nvcc on c01
  - set cmake variables: -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc -DNVSHMEM_PREFIX=/opt/nvshmem/ -DMPI_HOME=/usr/hpcx/ompi
- nvshmem library version 2.9.0 located at /opt/nvshmem/lib/ on c01
