import numpy as np
import pandas as pd
import sys
import os

# Output file
output = "tables/table_6.txt"
# Forecast types
ftypes = ["bic", "mavg"]
# Number of networks to use at each time step
n = 5

with open(output, "w") as f:
    pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from forecasting import compute_mape_df

# Compute the benchmark RMSE
avar = pd.read_csv("../../results/benchmarks/avar_forecasts.csv", index_col=0, header=[0,1])
avar.index = pd.to_datetime(avar.index)
benchmark = avar[["AvAR({2,13,25})"]]
# Replace columns, since the forecast horizon column headers have been loaded as stings
benchmark.columns = pd.MultiIndex.from_product([["AvAR({2,13,25})"], range(1, 13)])

for model in ["global", "standard", "local"]:
    # Compute averages and standard deviations for MAPEs
    mape_0 = pd.concat([pd.read_csv(f"../mapes/mape_df_{model}_{ftype}_{n}_0.csv", index_col=0) for ftype in ftypes], axis=1)
    mape_avg_df = mape_0.copy()
    mape_avg_df_sq = mape_0 ** 2
    for i in range(1, 100):
        mape_i = pd.concat([pd.read_csv(f"../mapes/mape_df_{model}_{ftype}_{n}_{i}.csv", index_col=0) for ftype in ftypes], axis=1)
        mape_avg_df += mape_i
        mape_avg_df_sq += mape_i ** 2

    # Average across the 100 runs
    mape_avg_df /= 100
    mape_std_df = np.sqrt((mape_avg_df_sq / 100) - (mape_avg_df ** 2))
    # MAPE
    mape_stats_df = np.round(mape_avg_df, 2).map(lambda x: f"{x:.2f}").add("\\tiny{$\\pm$").add(np.round(mape_std_df, 2).map(lambda x: f"{x:.2f}")).add("}")
    with open(output, "a") as f:
        print(f"Model: {model.capitalize()}", file=f)
        print(mape_stats_df.T[[1,3,6,9,12]].to_latex(), file=f)