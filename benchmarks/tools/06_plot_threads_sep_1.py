import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV data into a Pandas DataFrame
csv_file = 'results/06_put_granularity.csv'  # Updated file path
df = pd.read_csv(csv_file)

# Filter the data as per your conditions
filtered_df = df[(df['data_separation'].isin([0, 1])) & (df['grid_dim'] == 1) & (df['block_dim'].isin([1, 64, 512]))]

# Define a function to convert bytes to suitable units (kB, MB, GB, etc.)
def format_bytes(bytes):
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024**2:
        return f"{bytes / 1024:.2f} kB"
    elif bytes < 1024**3:
        return f"{bytes / (1024**2):.2f} MB"
    else:
        return f"{bytes / (1024**3):.2f} GB"

# Create a dictionary to store data for each line
line_data = {}

# Define a list of markers for different block dimensions
block_dim_markers = ['o', 's', '^', 'D', 'v']

# Iterate over distinct combinations of data_separation and block_dim
for sep in filtered_df['data_separation'].unique():
    block_dim_counter = 0
    for block_dim in filtered_df['block_dim'].unique():
        label = f"block_dim={block_dim}, sep={sep}"
        data = filtered_df[(filtered_df['data_separation'] == sep) & (filtered_df['block_dim'] == block_dim)]
        x_values = data['message_size'].apply(format_bytes)
        y_values = data['throughput'] / (1024**3)  # Convert throughput to GB
        line_data[label] = {'x_values': x_values, 'y_values': y_values, 'marker': block_dim_markers[block_dim_counter]}
        block_dim_counter += 1

# Create the plot
plt.figure(figsize=(12, 8))

colors = plt.cm.viridis(np.linspace(0, 1, (int)(len(line_data) / 2)))

for i, (label, data) in enumerate(line_data.items()):
    x_values = data['x_values']
    y_values = data['y_values']
    marker = data['marker']
    
    # Set linestyle to dashed for sep=0 and solid for sep=1
    linestyle = '--' if 'sep=0' in label else '-'
    
    plt.plot(x_values, y_values, label=label, color=colors[i % (int)(len(line_data) / 2)], marker=marker, linestyle=linestyle, linewidth=2.5)

# Customize the plot
plt.xlabel('Message Size', fontsize=25)
plt.ylabel('Throughput (GB)', fontsize=25)
plt.xticks(rotation=45, fontsize=23)
plt.yticks(fontsize=23)

# Make the legend slightly smaller
legend = plt.legend(fontsize=20)
for legend_handle in legend.legendHandles:
    legend_handle._sizes = [40]

# Show the plot
plt.tight_layout()
plt.grid(True)
plt.show()
