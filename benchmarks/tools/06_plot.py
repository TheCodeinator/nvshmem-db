import pandas as pd
import matplotlib.pyplot as plt
import os

filename = 'results/06_put_granularity.csv'

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

fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(30, 15))
fig.suptitle(f'06_put_granularity', fontsize=16)

for i, grid_size in enumerate(grid_sizes):
    for j, block_size in enumerate(block_sizes):
        ax = axs[i, j]
        ax.set_xscale('log', base=2)  # set x-axis to logarithmic scale

        for idx, node_count in enumerate(node_counts):
            df_subset = df[(df['num_hosts'] == node_count) & (df['grid_dim'] == grid_size) & (df['block_dim'] == block_size)]

            colors = ['g', 'b', 'r']
            label = node_labels.get(node_count, f"Node Count: {node_count}")  # use the dictionary to get the label
            df_subset.set_index('message_size')['throughput'].plot(ax=ax, marker='o', color=colors[idx], label=label)

        ax.set_title(f'Grid Size: {grid_size}, Block Size: {block_size}')
        ax.set_xlabel('Message size per thread (bytes)')
        ax.set_ylabel('Total throughput GB/s')
        ax.legend()

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()