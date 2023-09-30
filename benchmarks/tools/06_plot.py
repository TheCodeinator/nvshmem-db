import pandas as pd
import matplotlib.pyplot as plt
import os


def bytes_to_readable(num):
    """
    Converts bytes to a human-readable string format (e.g., 1KiB, 8MiB).
    """
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"


filename = 'results/07_sparse_sending.csv'

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    raise FileNotFoundError(f"Benchmark data '{os.path.basename(filename)}' missing, please use the bench.yaml "
                            f"ansible playbook")

node_counts = df['num_hosts'].unique()
grid_sizes = df['grid_dim'].unique()
block_sizes = df['block_dim'].unique()

node_labels = {
    0: "1 Node, NVLINK enabled",
    1: "1 Node, NVLINK disabled",
    2: "2 Nodes",
}

fig, axs = plt.subplots(nrows=grid_sizes.size, ncols=block_sizes.size, figsize=(30, 15))
fig.suptitle(f'06_put_granularity', fontsize=16)

global_min = df['message_size'].min()
global_max = df['message_size'].max()

print("Global min:", global_min)
print("Global max:", global_max)

for i, grid_size in enumerate(grid_sizes):
    for j, block_size in enumerate(block_sizes):
        ax = axs[i, j]
        ax.set_xscale('log', base=2)
        ax.set_xlim(global_min, global_max)

        for idx, node_count in enumerate(node_counts):
            df_subset = df[
                (df['num_hosts'] == node_count) & (df['grid_dim'] == grid_size) & (df['block_dim'] == block_size)]

            colors = ['g', 'b', 'r']
            label = node_labels.get(node_count, f"Node Count: {node_count}")
            df_subset.set_index('message_size')['throughput'].plot(ax=ax, marker='o', color=colors[idx], label=label)

        x_ticks = ax.get_xticks()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([bytes_to_readable(tick) for tick in x_ticks])

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.set_title(f'Grid Size: {grid_size}, Block Size: {block_size}')
        ax.set_xlabel('Message size per thread (bytes)')
        ax.set_ylabel('Total throughput GB/s')
        ax.legend()

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()
