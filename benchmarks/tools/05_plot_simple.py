import pandas as pd
import matplotlib.pyplot as plt
import os

filename = 'results/05_single_multi_launch_simple.csv'

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    raise FileNotFoundError(f"Benchmark data '{os.path.basename(filename)}' missing, please use the bench.yaml "
                            f"ansible playbook")

single_runtimes = df['time_single']
multi_runtimes = df['time_multi']
num_launches = df['launches']

# Match parameters of single vs multi thread benchmark
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 23})

plt.plot(num_launches, single_runtimes, color='red', label='Long running')
plt.plot(num_launches, multi_runtimes, color='blue', label='Short running')
plt.xlabel('Num launches (bytes)', fontsize=25)
plt.ylabel('Run time (s)', fontsize=25)

plt.legend(fontsize=25)

# Show the plot with a thicker grid
plt.grid(True, linewidth=1.2)

plt.title('Kernel Overhead', fontsize=30)

plt.tight_layout()
plt.show()
