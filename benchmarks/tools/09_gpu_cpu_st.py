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

# Create the plot
plt.figure(figsize=(10, 6))

# Plot CPU-driven data
plt.semilogx(data[data['is_gpu'] == 0]['message_size'], cpu_data['throughput'], label='CPU-driven', color='blue', marker='o')

# Plot nvshmem data
plt.semilogx(data[data['is_gpu'] == 1]['message_size'], nvshmem_data['throughput'], label='nvshmem', color='red', marker='s')

# Set x-axis labels as 2^log2(message_size)
plt.xticks(data['message_size'], labels=['$2^{%d}$' % x for x in data['log2_message_size']], rotation=45)

# Set axis labels and legend
plt.xlabel('Message Size')
plt.ylabel('Throughput')
plt.legend()

# Show the plot
plt.grid(True)
plt.title('Throughput Comparison')
plt.tight_layout()
plt.show()
