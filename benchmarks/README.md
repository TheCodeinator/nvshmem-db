
# Benchmarks

## CSV Scheme

type,node_count,in n,out n 

- type: Type of Benchmark (TBD)
- node_count: Number of nodes in the benchmark
- in (number): Input parameter(s) to the benchmark
- out (number): Output parameter(s) to the benchmark

## Benchmark IO

Extract dataframes from CSV files and plot them.

## Running Benchmarks

The ansible playbooks in this directory may be used to setup and execute benchmarks on a cluster of machines.
These playbooks are available:
- `sync.yaml`  : Sync the benchmarking code to each node.
- `compile.yaml`: Compile the benchmarking code on each node.
- `setup.yaml`: Set up the cluster for benchmarking. Runs `sync.yaml` and `compile.yaml`.
- `bench.yaml` : Run the benchmarking code on each node.
- `clean.yaml` : Remove all benchmarking code from each node.

These playbooks will use the inventory file `inventory.ini` as defined in the `ansible.cfg` file.
Note: Ansible playbooks are idempotent, so they may be run multiple times without issue.
They only perform the necessary actions to bring the system to the desired state.

The syntax for running an ansible playbook is:

```bash
ansible-playbook -u <user> <playbook>
```

For example, to run the setup playbook:

```bash
ansible-playbook -u johnsmith setup.yaml
```
