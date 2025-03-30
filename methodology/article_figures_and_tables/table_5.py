import numpy as np
import pandas as pd
import sys
import os

# Output file
output = "tables/table_5.txt"
# Forecast types
ftypes = ["bic", "mavg"]
# Number of networks to use at each time step
n = 5

with open(output, "w") as f:
    pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bank_of_england import rmse_vs_bank

# Compute the benchmark RMSE
avar = pd.read_csv("../../results/benchmarks/avar_forecasts.csv", index_col=0, header=[0,1])
avar.index = pd.to_datetime(avar.index)
benchmark = avar[["AvAR({2,13,25})"]]
# Replace columns, since the forecast horizon column headers have been loaded as stings
benchmark.columns = pd.MultiIndex.from_product([["AvAR({2,13,25})"], range(1, 13)])
benchmark_rmse = np.round(rmse_vs_bank(benchmark), 2).map(lambda x: f"{x:.2f}")

with open(output, "a") as f:
    print(f"Model: Benchmarks", file=f)
    print(benchmark_rmse.T.to_latex(), file=f)

for model in ["global", "standard", "local"]:
    # Drop the Bank of England column since this does not change across runs
    bank_rmse_0 = pd.concat([pd.read_csv(f"../rmses/rmse_df_{model}_{ftype}_bank_{n}_0.csv", index_col=0).drop(columns=["Bank of England"]) for ftype in ftypes], axis=1)
    bank_rmse_avg_df = bank_rmse_0.copy()
    bank_rmse_avg_df_sq = bank_rmse_0 ** 2
    for i in range(1, 100):
        bank_rmse_i = pd.concat([pd.read_csv(f"../rmses/rmse_df_{model}_{ftype}_bank_{n}_{i}.csv", index_col=0).drop(columns=["Bank of England"]) for ftype in ftypes], axis=1)
        bank_rmse_avg_df += bank_rmse_i
        bank_rmse_avg_df_sq += bank_rmse_i ** 2

    bank_rmse_avg_df /= 100
    bank_rmse_std_df = np.sqrt((bank_rmse_avg_df_sq / 100) - (bank_rmse_avg_df ** 2))
    # Comparison to Bank of England
    bank_stats_df = np.round(bank_rmse_avg_df, 2).map(lambda x: f"{x:.2f}").add("\\tiny{$\\pm$").add(np.round(bank_rmse_std_df, 2).map(lambda x: f"{x:.2f}")).add("}")

    with open(output, "a") as f:
        print(f"Model: {model.capitalize()}", file=f)
        print(bank_stats_df.T.to_latex(), file=f)