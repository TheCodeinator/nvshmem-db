# nvshmem-db

Tested Environment:
- Compiler: /usr/bin/gcc-11 and /usr/bin/g++-11 on c01
- cuda compiler version: /usr/local/cuda-12/bin/nvcc on c01
- cmake cli variables: -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc -DNVSHMEM_PREFIX=/opt/nvshmem/ -DMPI_HOME=/usr/hpcx/ompi -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-11 -DCMAKE_BUILD_TYPE=Debug
- nvshmem library version 2.9.0 located at /opt/nvshmem/lib/ on c01
