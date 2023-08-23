import pandas as pd
import matplotlib.pyplot as plt
import os

filename = 'results/06_put_granularity.csv'

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    raise FileNotFoundError(f"Benchmark data '{os.path.basename(filename)}' missing, please use the bench.yaml "
                            f"ansible playbook")

# select desired grid and block sizes
grid_sizes = [1, 2, 4, 8]
block_sizes = [1, 8, 32, 64]

# get unique node counts for different grids
node_counts = df['num_hosts'].unique()

# create separate 4x4 grid for each node count
for node_count in node_counts:
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(30, 12))
    fig.suptitle(f'06_put_granularity, node count: {node_count}', fontsize=16)

    # loop over each subplot
    for i, grid_size in enumerate(grid_sizes):
        for j, block_size in enumerate(block_sizes):
            ax = axs[i, j]
            ax.set_xscale('log', base=2)  # set x-axis to logarithmic scale

            # subset dataframe by current node count, grid and block sizes
            df_subset = df[(df['num_hosts'] == node_count) & (df['grid_dim'] == grid_size) & (
                    df['block_dim'] == block_size)]

            # plot with single color for all elements in this subplot
            df_subset.set_index('message_size')['throughput'].plot(ax=ax, marker='o', color='b')

            ax.set_title(f'Grid Size: {grid_size}, Block Size: {block_size}')
            ax.set_xlabel('Number of Bytes')
            ax.set_ylabel('Throughput GB/s')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # adjust suptitle
    plt.show()
