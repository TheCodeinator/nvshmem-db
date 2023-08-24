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


filename = 'results/06_put_granularity.csv'

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    raise FileNotFoundError(f"Benchmark data '{os.path.basename(filename)}' missing, please use the bench.yaml "
                            f"ansible playbook")

# Filter the data for node_count=2
df = df[df['num_hosts'] == 2]

block_sizes = df['block_dim'].unique()
data_separations = [0, 1, 512]
styles = {
    0: ('-', 'o'),
    1: (':', 's'),
    512: ('--', '^')
}

# Create the 2x2 grid of plots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

# Specific grid sizes we want to focus on
grid_sizes_focus = [1, 2, 4, 8]

for ax, grid_size in zip(axs.ravel(), grid_sizes_focus):
    ax.set_xscale('log', base=2)

    # Plot each block_size configuration as a line for the current grid size
    for block_size in block_sizes:
        color = next(ax._get_lines.prop_cycler)['color']  # Get next color in the cycle
        for sep, style in zip(data_separations, styles.items()):
            df_subset = df[(df['grid_dim'] == grid_size) &
                           (df['block_dim'] == block_size) &
                           (df['data_separation'] == sep)]
            if not df_subset.empty:  # Ensure there's data to plot
                df_subset.set_index('message_size')['throughput'].plot(ax=ax,
                                                                       color=color,
                                                                       linestyle=style[1][0],
                                                                       marker=style[1][1],
                                                                       label=f'Block Size: {block_size}, Data Separation: {sep}')

    x_ticks = ax.get_xticks()
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([bytes_to_readable(tick) for tick in x_ticks])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_title(f'Sparse Sending for Grid Size: {grid_size}')
    ax.set_xlabel('Message size (bytes)')
    ax.set_ylabel('Total throughput GB/s')
    ax.legend()

plt.tight_layout()
plt.show()
