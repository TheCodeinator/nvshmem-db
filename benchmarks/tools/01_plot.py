import pandas as pd
import matplotlib.pyplot as plt

# read the csv
df = pd.read_csv('results/01_put_coalescing.csv')

# select desired grid and block sizes
grid_sizes = [1, 8, 64]
block_sizes = [1, 8, 64]

# get unique node counts for different grids
node_counts = df['node_count'].unique()

# create separate 3x3 grid for each node count
for node_count in node_counts:
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    fig.suptitle(f'Node Count: {node_count}', fontsize=16)

    # loop over each subplot
    for i, grid_size in enumerate(grid_sizes):
        for j, block_size in enumerate(block_sizes):
            ax = axs[i, j]

            # subset dataframe by current node count, grid and block sizes
            df_subset = df[(df['node_count'] == node_count) & (df['in_num_grids'] == grid_size) & (df['in_num_blocks'] == block_size)]

            # plot with single color for all elements in this subplot
            df_subset.set_index('in_num_elements')['out_throughput_one_thread_sep'].plot(ax=ax, marker='o', color='b', label='one thread sep')
            df_subset.set_index('in_num_elements')['out_throughput_one_thread_once'].plot(ax=ax, marker='o', color='r', label='one thread once')
            df_subset.set_index('in_num_elements')['out_throughput_multi_thread_sep'].plot(ax=ax, marker='o', color='g', label='multi thread sep')

            ax.set_title(f'Grid Size: {grid_size}, Block Size: {block_size}')
            ax.set_xlabel('Number of Elements')
            ax.set_ylabel('Throughput')
            ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # adjust suptitle
    plt.show()
