import pandas as pd
import matplotlib.pyplot as plt
import os

filename = 'results/05_single_multi_launch.csv'

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    raise FileNotFoundError(f"Benchmark data '{os.path.basename(filename)}' missing, please use the bench.yaml "
                            f"ansible playbook")
