# When working on the project in the TUD infrastructure, the following code can be used to load the necessary libraries
# It might have to be slightly adjusted
# We recommend adding this to your ~/.bashrc file

# specify development environment to load
DEV_ENV="NVSHMEM_DB"

if [[ "$DEV_ENV" == "NVSHMEM_DB" ]]; then
        echo "loading development environment NVSHMEM_DB"

        export NVSHMEM_HOME='/opt/nvshmem'
        export CUDA_HOME='/usr/local/cuda-12.0'
        export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
        export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/gcc-11'
        export LD_LIBRARY_PATH=/usr/hpcx/ompi/lib/:$LD_LIBRARY_PATH
        export HYDRA_HOME=/opt/hydra
        export PATH=$HYDRA_HOME/bin:$CUDA_HOME/bin:$PATH
        export HPCX_HOME="/usr/hpcx/"
        source ~/software/modules-4.5.0/init/bash
        module use $HPCX_HOME/modulefiles
        module load hpcx-mt
fi
