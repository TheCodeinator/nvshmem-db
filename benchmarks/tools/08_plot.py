import pandas as pd
import matplotlib.pyplot as plt
import os
import math

filename = 'results/08_tuple_scan.csv'

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    raise FileNotFoundError(f"Benchmark data '{os.path.basename(filename)}' missing")

unique_grid_sizes = df['grid_dim'].unique()
unique_multipliers = df['send_buffer_size_multiplier'].unique()
unique_tuple_sizes = df['tuple_size'].unique()

grid_dim = math.ceil(math.sqrt(len(unique_grid_sizes)))

fig, axs = plt.subplots(grid_dim, grid_dim, figsize=(20, 10))
axs = axs.flatten()

legend_handles = []

for i, grid_size in enumerate(unique_grid_sizes):
    df_grid = df[df['grid_dim'] == grid_size]

    for multiplier in unique_multipliers:
        for tuple_size in unique_tuple_sizes:
            df_filtered = df_grid[
                (df_grid['send_buffer_size_multiplier'] == multiplier) & (df_grid['tuple_size'] == tuple_size)]
            line, = axs[i].plot(df_filtered['block_dim'], df_filtered['throughput_gb_s'], marker='o',
                                label=f"Multiplier {multiplier}, Tuple Size {tuple_size}")
            axs[i].set_title(f"Grid Size: {grid_size}")
            axs[i].set_xlabel("Block Dim")
            axs[i].set_ylabel("Throughput (GB/s)")

            if i == 0:
                legend_handles.append(line)

for i in range(len(unique_grid_sizes), grid_dim * grid_dim):
    fig.delaxes(axs[i])

fig.legend(handles=legend_handles, labels=[handle.get_label() for handle in legend_handles], loc='lower right')

plt.tight_layout()
plt.show()
