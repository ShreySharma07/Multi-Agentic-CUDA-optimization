import pandas as pd

file_path = 'profile_results.csv'

# 1. Dynamically find where the CSV headers start
header_row = 0
with open(file_path, 'r') as f:
    for i, line in enumerate(f):
        if line.startswith('"ID"') or line.startswith('"Index"'):
            header_row = i
            break

# 2. Load the CSV
df = pd.read_csv(file_path, skiprows=header_row)

# 3. Filter for your vectorAdd kernel (using partial string match)
kernel_df = df[df['Kernel Name'].str.contains('vectorAdd', na=False)]

# 4. Filter for the exact metric names found in your file
metrics_to_check = [
    'Compute (SM) Throughput',
    'Memory Throughput',
    'DRAM Throughput',
    'L1/TEX Cache Throughput',
    'L2 Cache Throughput',
    'Registers Per Thread'
]

filtered_metrics = kernel_df[kernel_df['Metric Name'].isin(metrics_to_check)]

# Drop any empty rows just in case, and print the results cleanly
filtered_metrics = filtered_metrics.dropna(subset=['Metric Name'])
print(filtered_metrics[['Metric Name', 'Metric Value', 'Metric Unit']].to_string(index=False))