import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data from CSV file
data = pd.read_csv("results/09_gpu_cpu_st.csv")

# Separate data for is_gpu=0 and is_gpu=1
cpu_data = data[data['is_gpu'] == 0]
nvshmem_data = data[data['is_gpu'] == 1]

# Create a new column with log2(message_size)
data['log2_message_size'] = np.log2(data['message_size'])

# Create the plot with larger font sizes and line widths
plt.figure(figsize=(12, 8))  # Increase the figure size
plt.rcParams.update({'font.size': 23})  # Increase the font size

# Round message_size to the nearest power of 2
rounded_message_sizes = [2**int(np.round(np.log2(size))) for size in cpu_data['message_size']]

# Plot CPU-driven data with thicker lines
plt.semilogx(rounded_message_sizes, cpu_data['throughput'], label='CPU initiated', color='blue', marker='o', linewidth=3)

# Plot nvshmem data with thicker lines
plt.semilogx(rounded_message_sizes, nvshmem_data['throughput'], label='GPU-initiated', color='red', marker='s', linewidth=3)

# Set x-axis labels as powers of 2 bytes, MB, or GB
x_tick_locations = np.unique(rounded_message_sizes)
x_labels = []

for size in x_tick_locations:
    if size < 1024:
        x_labels.append(f'{int(size)}B')
    elif size < 1024 * 1024:
        x_labels.append(f'{int(size / 1024)}KB')
    else:
        x_labels.append(f'{int(size / (1024 * 1024))}GB')

plt.xticks(x_tick_locations, labels=x_labels, rotation=45)

# Set axis labels and legend
plt.xlabel('Message Size', fontsize=25)
plt.ylabel('Throughput', fontsize=25)
plt.legend(fontsize=25)

# Show the plot with a thicker grid
plt.grid(True, linewidth=1.2)

# Set a larger title font size
plt.title('Throughput Comparison', fontsize=30)

plt.tight_layout()
plt.show()
