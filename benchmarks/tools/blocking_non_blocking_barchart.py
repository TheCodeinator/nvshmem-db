import matplotlib.pyplot as plt

# Given values in nanoseconds
nbi_quiet_all = 5655437773
nbi_quiet_each = 5676148861
nbi_no_quiet = 5673918505
blocking = 5674057820

# Convert values to seconds
nbi_quiet_all_seconds = nbi_quiet_all / 1e9
nbi_quiet_each_seconds = nbi_quiet_each / 1e9
nbi_no_quiet_seconds = nbi_no_quiet / 1e9
blocking_seconds = blocking / 1e9

# Names for the bars
labels = ['NBI quiet all', 'nbi quiet each', 'nbi no quiet', 'blocking']

# Values in seconds
values = [nbi_quiet_all_seconds, nbi_quiet_each_seconds, nbi_no_quiet_seconds, blocking_seconds]

# Create a bar chart
plt.bar(labels, values)

# Set the title and labels with increased font size
plt.title('Comparison of NVSHMEM blocking and non-blocking interface', fontsize=20)
plt.xlabel('Execution Mode', fontsize=18)
plt.ylabel('Time (seconds)', fontsize=18)

# Set the font size for x-axis labels
plt.xticks(fontsize=16)

# Show the plot
plt.show()
