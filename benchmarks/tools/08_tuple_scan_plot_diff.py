import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# Read the CSV data into a Pandas DataFrame
data = pd.read_csv('results/08_tuple_scan.csv')

# this tuple size has the best results and is the most realistic
data = data[data['tuple_size'] == 1024]

data_atomic = data[data['offset_mode'] == 0].reset_index(drop=True)
data_syncfree = data[data['offset_mode'] == 1].reset_index(drop=True)

# Filter the DataFrame to include only rows with offset_mode == 0
filtered_data = data_atomic
filtered_data.loc[:, 'throughput_gb_s'] = data_atomic['throughput_gb_s'] - data_syncfree['throughput_gb_s']

# Get the minimum and maximum throughput values
min_throughput = filtered_data['throughput_gb_s'].min()
max_throughput = filtered_data['throughput_gb_s'].max()

# Create a color gradient based on throughput_gb_s values
norm = plt.Normalize(vmin=min_throughput, vmax=max_throughput)
colors = cm.viridis(norm(filtered_data['throughput_gb_s']))

# Create the 3D plot with adjusted figure size
fig = plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
ax = fig.add_subplot(111, projection='3d')

# Set axis labels
ax.set_xlabel('Block Dim')
ax.set_ylabel('Grid Dim')
ax.set_zlabel('Throughput GB/s')

# Create a grid of block_dim and grid_dim values
block_dim_values = filtered_data['block_dim'].unique()
grid_dim_values = filtered_data['grid_dim'].unique()
block_dim_mesh, grid_dim_mesh = np.meshgrid(block_dim_values, grid_dim_values)

# Reshape throughput_gb_s to match the dimensions of the grid
throughput_gb_s_mesh = filtered_data['throughput_gb_s'].values.reshape(len(grid_dim_values), len(block_dim_values))

# Create the plane plot
plane = ax.plot_surface(block_dim_mesh, grid_dim_mesh, throughput_gb_s_mesh,
                        cmap='viridis', facecolors=cm.viridis(norm(throughput_gb_s_mesh)))

# Set the title
plt.title('Diff: Atomic Increment - Sync-free')

# Show the plot
plt.show()
