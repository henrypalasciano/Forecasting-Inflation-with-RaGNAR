import numpy as np
import pandas as pd
import sys
import os

# Output file
output = "tables/table_2.txt"
# Forecast type
ftype = "bic"
# Number of networks to use at each time step
n = 1

with open(output, "w") as f:
    pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from forecasting import compute_rmse_df

# Compute the benchmark RMSE
avar = pd.read_csv("../../results/benchmarks/avar_forecasts.csv", index_col=0, header=[0,1])
avar.index = pd.to_datetime(avar.index)
benchmark = avar[["AvAR({2,13,25})"]]
# Replace columns, since the forecast horizon column headers have been loaded as stings
benchmark.columns = pd.MultiIndex.from_product([["AvAR({2,13,25})"], range(1, 13)])
benchmark_rmse = compute_rmse_df(benchmark).to_numpy().reshape(-1, 1)

for model in ["global", "standard", "local"]:
    # Compute averages and standard deviations for RMSEs
    rmse_0 = pd.read_csv(f"../rmses/rmse_df_{model}_{ftype}_{n}_0.csv", index_col=0)
    rmse_avg_df = rmse_0.copy()
    rmse_avg_df_sq = rmse_0 ** 2
    for i in range(1, 100):
        rmse_i = pd.read_csv(f"../rmses/rmse_df_{model}_{ftype}_{n}_{i}.csv", index_col=0)
        rmse_avg_df += rmse_i
        rmse_avg_df_sq += rmse_i ** 2

    rmse_avg_df /= 100
    rmse_std_df = np.sqrt((rmse_avg_df_sq / 100) - (rmse_avg_df ** 2))
    rel_stats_df = np.round(rmse_avg_df / benchmark_rmse, 2).map(lambda x: f"{x:.2f}").add("\\tiny{$\\pm$").add(np.round(rmse_std_df / benchmark_rmse, 2).map(lambda x: f"{x:.2f}")).add("}")
    with open(output, "a") as f:
        print(f"Model: {model.capitalize()}", file=f)
        print(rel_stats_df.T[[1,3,6,9,12]].to_latex(), file=f)