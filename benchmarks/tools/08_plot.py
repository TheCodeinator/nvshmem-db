import pandas as pd
import matplotlib.pyplot as plt
import os
import math


def bytes_to_readable(num):
    """
    Converts bytes to a human-readable string format (e.g., 1KiB, 8MiB).
    """
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"


filename = 'results/08_tuple_scan.csv'

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    raise FileNotFoundError(f"Benchmark data '{os.path.basename(filename)}' missing, please use the bench.yaml "
                            f"ansible playbook")

# Unique grid sizes, send_buffer_size_multiplier values, and tuple sizes
unique_grid_sizes = df['grid_dim'].unique()
unique_multipliers = df['send_buffer_size_multiplier'].unique()
unique_tuple_sizes = df['tuple_size'].unique()

# Calculate the dimensions of the 2D grid for subplots
grid_dim = math.ceil(math.sqrt(len(unique_grid_sizes)))

# Create subplots
fig, axs = plt.subplots(grid_dim, grid_dim, figsize=(20, 10))

# Flatten the axs array for easier indexing
axs = axs.flatten()

show_legend = False

# Loop over each subplot and plot data
for i, grid_size in enumerate(unique_grid_sizes):
    df_grid = df[df['grid_dim'] == grid_size]

    for multiplier in unique_multipliers:
        for tuple_size in unique_tuple_sizes:
            df_filtered = df_grid[
                (df_grid['send_buffer_size_multiplier'] == multiplier) & (df_grid['tuple_size'] == tuple_size)]
            axs[i].plot(df_filtered['block_dim'], df_filtered['throughput_gb_s'], marker='o',
                        label=f"Multiplier {multiplier}, Tuple Size {tuple_size}")

    axs[i].set_title(f"Grid Size: {grid_size}")
    axs[i].set_xlabel("Block Dim")
    axs[i].set_ylabel("Throughput (GB/s)")
    if show_legend:
        axs[i].legend()

for i in range(len(unique_grid_sizes), grid_dim * grid_dim):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.show()
