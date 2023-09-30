import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

# Read the CSV data into a Pandas DataFrame
data = pd.read_csv("results/11_kernel_launch_overhead.csv")

# Increase font size for labels and ticks
font_prop = font_manager.FontProperties(size=14)

# Create the plot
plt.figure(figsize=(10, 6))  # Set the figure size
plt.plot(data['n_launches'], data['ex_time'], color='blue', marker='o', linewidth=2.3)  # Plot the data

# Set axis labels with increased font size
plt.xlabel('number of kernel launches', fontproperties=font_prop)
plt.ylabel('execution time (s)', fontproperties=font_prop)

# Set title with increased font size
plt.title('Kernel Launch Overhead', fontproperties=font_prop)

# Show the plot
plt.show()
