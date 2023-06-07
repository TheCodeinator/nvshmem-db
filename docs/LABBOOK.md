# Project History

This is a living document for issues during the course of the project, document time of the issue and solution.

### 07.06.2023 MPI not installed, not found in cmake 

=> is installed, but in nonstandard location /usr/hpcx/ompi, set -DMPI_HOME=/usr/hpcx/ompi

### 07.06.2023 A couple of paths must be set in bash to find libraries correctly

=> Append the contents of the file ROOT/docs/dev_env.sh to your ~/.bashrc file 
