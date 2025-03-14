import numpy as np
import pandas as pd
import os

from forecasting import compute_rmse_df

# Create folders for storing the averaged results
output_folder = "average_results"
os.makedirs(output_folder, exist_ok=True)

# Text files to write the averaged RMSE results
avg_rmse = os.path.join(output_folder, "avg_rmse.txt") # Average across networks
bic_rmse = os.path.join(output_folder, "bic_rmse.txt") # BIC forecasts
avgnar_rmse = os.path.join(output_folder, "mavg_rmse.txt") # Average across models
# Text files to write the averaged MAPE results
mape = os.path.join(output_folder, "mape.txt")
# Create the text files
with open(avg_rmse, "w") as f:
    pass
with open(bic_rmse, "w") as f:
    pass
with open(avgnar_rmse, "w") as f:
    pass
with open(mape, "w") as f:
    pass

avar = pd.read_csv("../results/benchmarks/avar_forecasts.csv", index_col=0, header=[0,1])
avar.index = pd.to_datetime(avar.index)
benchmark = avar[["AvAR({2,13,25})"]]
# Replace columns, since the forecast horizon column headers have been loaded as stings
benchmark.columns = pd.MultiIndex.from_product([["AvAR({2,13,25})"], range(1, 13)])
benchmark_rmse = compute_rmse_df(benchmark).to_numpy().reshape(-1, 1)

# Loop through model classes
for model in ["global", "standard", "local"]:
    # Loop through forecast tipes
    for ftype in ["avg", "bic", "mavg"]:
        # Loop through number of best networks
        for j in [1, 2, 5]:
            # Compute averages and standard deviations for RMSEs
            rmse_0 = pd.read_csv(f"rmses/rmse_df_{model}_{ftype}_{j}_0.csv", index_col=0)
            rmse_avg_df = rmse_0.copy()
            rmse_avg_df_sq = rmse_0 ** 2
            # Drop the Bank of England column since this does not change across runs
            bank_rmse_0 = pd.read_csv(f"rmses/rmse_df_{model}_{ftype}_bank_{j}_0.csv", index_col=0).drop(columns=["Bank of England"])
            bank_rmse_avg_df = bank_rmse_0.copy()
            bank_rmse_avg_df_sq = bank_rmse_0 ** 2
            # Loop through the 100 RaGNAR runs
            for i in range(1, 100):
                rmse_i = pd.read_csv(f"rmses/rmse_df_{model}_{ftype}_{j}_{i}.csv", index_col=0)
                rmse_avg_df += rmse_i
                rmse_avg_df_sq += rmse_i ** 2
                bank_rmse_i = pd.read_csv(f"rmses/rmse_df_{model}_{ftype}_bank_{j}_{i}.csv", index_col=0).drop(columns=["Bank of England"])
                bank_rmse_avg_df += bank_rmse_i
                bank_rmse_avg_df_sq += bank_rmse_i ** 2
            # Average across the 100 runs
            rmse_avg_df /= 100
            rmse_std_df = np.sqrt((rmse_avg_df_sq / 100) - (rmse_avg_df ** 2))
            # Absolute RMSE
            abs_stats_df = np.round(rmse_avg_df, 2).map(lambda x: f"{x:.2f}").add("\\tiny{$\\pm$").add(np.round(rmse_std_df, 2).map(lambda x: f"{x:.2f}")).add("}")
            # Relative RMSE
            rel_stats_df = np.round(rmse_avg_df / benchmark_rmse, 2).map(lambda x: f"{x:.2f}").add("\\tiny{$\\pm$").add(np.round(rmse_std_df / benchmark_rmse, 2).map(lambda x: f"{x:.2f}")).add("}")
            bank_rmse_avg_df /= 100
            bank_rmse_std_df = np.sqrt((bank_rmse_avg_df_sq / 100) - (bank_rmse_avg_df ** 2))
            # Comparison to Bank of England
            bank_stats_df = np.round(bank_rmse_avg_df, 2).map(lambda x: f"{x:.2f}").add("\\tiny{$\\pm$").add(np.round(bank_rmse_std_df, 2).map(lambda x: f"{x:.2f}")).add("}")
            # Write the results to a text file
            output_file = f"{output_folder}/{ftype}_rmse.txt"
            with open(output_file, "a") as f:
                print(f"Model: {model}, N: {j}", file=f)
                print("Absolute RMSE:", file=f)
                print(abs_stats_df.T[[1,3,6,9,12]].to_latex(), file=f)
                print("Relative RMSE:", file=f)
                print(rel_stats_df.T[[1,3,6,9,12]].to_latex(), file=f)
                print("Comparison to Bank of England:", file=f)
                print(bank_stats_df.T.to_latex(), file=f)
                print("=" * 100, "\n", file=f)
    # For the MAPE only interested in j = 5, forecasts using BIC and AvGNAR, and absolute RMSEs
    output_file = f"{output_folder}/mape.txt"
    # Loop through forecast types
    for ftype in ["bic", "mavg"]:
        # Compute averages and standard deviations for MAPEs
        mape_0 = pd.read_csv(f"mapes/mape_df_{model}_{ftype}_5_0.csv", index_col=0)
        mape_avg_df = mape_0.copy()
        mape_avg_df_sq = mape_0 ** 2
        bank_mape_0 = pd.read_csv(f"mapes/mape_df_{model}_{ftype}_bank_5_0.csv", index_col=0).drop(columns=["Bank of England"])
        bank_mape_avg_df = bank_mape_0.copy()
        bank_mape_avg_df_sq = bank_mape_0 ** 2
        # Loop through the 100 RaGNAR runs
        for i in range(1, 100):
            mape_i = pd.read_csv(f"mapes/mape_df_{model}_{ftype}_5_{i}.csv", index_col=0)
            mape_avg_df += mape_i
            mape_avg_df_sq += mape_i ** 2
            bank_mape_i = pd.read_csv(f"mapes/mape_df_{model}_{ftype}_bank_5_{i}.csv", index_col=0).drop(columns=["Bank of England"])
            bank_mape_avg_df += bank_mape_i
            bank_mape_avg_df_sq += bank_mape_i ** 2
        # Average across the 100 runs
        mape_avg_df /= 100
        mape_std_df = np.sqrt((mape_avg_df_sq / 100) - (mape_avg_df ** 2))
        # MAPE
        mape_stats_df = np.round(mape_avg_df, 2).map(lambda x: f"{x:.2f}").add("\\tiny{$\\pm$").add(np.round(mape_std_df, 2).map(lambda x: f"{x:.2f}")).add("}")
        bank_mape_avg_df /= 100
        bank_mape_std_df = np.sqrt((bank_mape_avg_df_sq / 100) - (bank_mape_avg_df ** 2))
        # Comparison to Bank of England
        bank_mape_stats_df = np.round(bank_mape_avg_df, 2).map(lambda x: f"{x:.2f}").add("\\tiny{$\\pm$").add(np.round(bank_mape_std_df, 2).map(lambda x: f"{x:.2f}")).add("}")
        # Write the results to a text file
        with open(output_file, "a") as f:
            print(f"Model: {model}, Type: {ftype}", file=f)
            print("MAPE:", file=f)
            print(mape_stats_df.T[[1,3,6,9,12]].to_latex(), file=f)
            print("Comparison to Bank of England:", file=f)
            print(bank_mape_stats_df.T.to_latex(), file=f)
            print("=" * 100, "\n", file=f)