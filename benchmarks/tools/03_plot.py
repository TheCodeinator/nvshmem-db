import pandas as pd
import matplotlib.pyplot as plt
import os

filename = 'results/03_packet_size_put_nbi.csv'

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    raise FileNotFoundError(f"Benchmark data '{os.path.basename(filename)}' missing, please use the bench.yaml "
                            f"ansible playbook")

# select desired grid and block sizes
grid_sizes = [1, 8, 64]
block_sizes = [1, 8, 64]

# get unique node counts for different grids
node_counts = df['node_count'].unique()

# create separate 3x3 grid for each node count
for node_count in node_counts:
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    fig.suptitle(f'03_packet_size_put_nbi, node count: {node_count}', fontsize=16)

    # loop over each subplot
    for i, grid_size in enumerate(grid_sizes):
        for j, block_size in enumerate(block_sizes):
            ax = axs[i, j]
            ax.set_xscale('log', base=2)  # set x-axis to logarithmic scale

            # subset dataframe by current node count, grid and block sizes
            df_subset = df[(df['node_count'] == node_count) & (df['in_num_grids'] == grid_size) & (
                    df['in_num_blocks'] == block_size)]

            # plot with single color for all elements in this subplot
            df_subset.set_index('in_num_bytes')['out_throughput'].plot(ax=ax, marker='o', color='b')

            ax.set_title(f'Grid Size: {grid_size}, Block Size: {block_size}')
            ax.set_xlabel('Number of Bytes')
            ax.set_ylabel('Throughput GB/s')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # adjust suptitle
    plt.show()
