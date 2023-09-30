import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV data
df = pd.read_csv('results/06_put_granularity.csv')

grid_dim = 1

# Filter the data
filtered_df = df[(df['grid_dim'] == grid_dim)
                 & (df['data_separation'] == 0)
                 & (df['block_dim'].isin([1, 32, 64, 512]))
                 & (df['num_hosts'] == 2)]

# Sort data by block_dim
filtered_df.sort_values(['block_dim', 'message_size'], inplace=True)
filtered_df.reset_index(inplace=True)

filtered_df_message_str = filtered_df

def format_bytes(bytes):
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024**2:
        return f"{bytes / 1024:.0f} KiB"
    elif bytes < 1024**3:
        return f"{bytes / (1024**2):.0f} MiB"
    else:
        return f"{bytes / (1024**3):.0f} GiB"

# Group data by block_dim and create a plot
block_dims = filtered_df['block_dim'].unique()
# colors = plt.cm.viridis(np.linspace(0, 1, len(block_dims)))
colors = ['red', 'blue', 'green', 'gray']
markers = ['o', 's', '^', 'D', 'v', 'p', '>', '<', 'h', '+']

plt.figure(figsize=(10, 8))

for i, block_dim in enumerate(block_dims):
    block_df = filtered_df[filtered_df['block_dim'] == block_dim]
    plt.plot(block_df['message_size'].apply(format_bytes), block_df['throughput'], label=f'{grid_dim} block(s) of {block_dim} thread(s)',
             marker=markers[i], color=colors[i], linewidth=2, markersize=8)

# Set plot properties
plt.title('Throughput vs. Message Size', fontsize=25)
plt.xlabel('Message Size', fontsize=20)
plt.ylabel('Throughput (GB/s)', fontsize=20)
plt.legend(fontsize=14)

# Show every 2nd x-axis tick
x_ticks = plt.xticks()[0]
plt.xticks(x_ticks[::2], fontsize=18)


plt.yticks(fontsize=18)

plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
