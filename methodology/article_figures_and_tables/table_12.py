import numpy as np
import pandas as pd
import sys
import os

# Output file
output = "tables/table_12.txt"

with open(output, "w") as f:
    pass

for model in ["global", "standard", "local"]:
    # Compute averages and standard deviations for RMSEs
    rmse_0 = pd.read_csv(f"../dir_results/rmse_df_{model}_0.csv", index_col=0)
    rmse_avg_df = rmse_0.copy()
    rmse_avg_df_sq = rmse_0 ** 2
    for i in range(1, 100):
        rmse_i = pd.read_csv(f"../dir_results/rmse_df_{model}_{i}.csv", index_col=0)
        rmse_avg_df += rmse_i
        rmse_avg_df_sq += rmse_i ** 2

    rmse_avg_df /= 100
    rmse_std_df = np.sqrt((rmse_avg_df_sq / 100) - (rmse_avg_df ** 2))
    rmse_stats_df = np.round(rmse_avg_df, 2).map(lambda x: f"{x:.2f}").add("\\tiny{$\\pm$").add(np.round(rmse_std_df, 2).map(lambda x: f"{x:.2f}")).add("}")
    with open(output, "a") as f:
        print(f"Model: {model.capitalize()}", file=f)
        print(rmse_stats_df.T[[1,3,6,9,12]].to_latex(), file=f)