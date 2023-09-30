import pandas as pd
import matplotlib.pyplot as plt
import os

filename = 'results/05_single_multi_launch.csv'

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    raise FileNotFoundError(f"Benchmark data '{os.path.basename(filename)}' missing, please use the bench.yaml "
                            f"ansible playbook")

cpu_runtimes = df['time_rdma']
gpu_runtimes = df['time_nvshmem']
num_launches = df['launches']
send_size = df['num_bytes']

# Match parameters of single vs multi thread benchmark
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 23})

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.semilogx(send_size, cpu_runtimes, label='CPU driven')
ax1.semilogx(send_size, gpu_runtimes, label='GPU driven')
ax1.xlabel('Size total (bytes)', fontsize=25)
ax1.ylabel('Run time (s)', fontsize=25)

# double use y to show number of kernel launches for cpu-driven rdma
ax2.semilogx(send_size, num_launches, label='Kernel launches')
ax2.ylabel('Launches', fontsize=25)

plt.legend(fontsize=25)

# Show the plot with a thicker grid
plt.grid(True, linewidth=1.2)

plt.title('CPU vs. GPU driven communication', fontsize=30)

plt.tight_layout()
plt.show()
