import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sns.set(style="whitegrid")

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ar_forecasts import rolling_bic_ar_forecast
from scoring import cpi_rolling_mse
from bic_gnar import select_bic_model_order

# Load the data and calculate the inflation rate
cpi_monthly_data = pd.read_csv("../data/cpi_monthly_data.csv", index_col=0)
cpi = cpi_monthly_data[["00"]]
inflation_rate = cpi.pct_change(12).dropna(how="all") * 100
inflation_rate.columns = ["Inflation Rate"]
inflation_rate.index = pd.to_datetime(inflation_rate.index)

# Compute the BIC for the AR model
bic_ar, bic_p = rolling_bic_ar_forecast(inflation_rate, 240, 12, "2009-12-01", "2024-11-01")
dates = bic_ar.index

# Load the squared error dataframe
se_df_list = []
# Stored in 10 different files so as to not exceed the 100MB limit on GitHub - can replace this code with a single file if necessary
for i in range(1, 11):
    # Replace the path with that of the results of a different run of RaGNAR if necessary
    se_df_list.append(pd.read_csv(f"../../results/ragnar/se_df/se_df_{i}.csv", index_col=0, header=[0, 1]))
se_df = pd.concat(se_df_list)
# Reconstruct the MultiIndex with the dates in the correct format
se_df.columns = pd.MultiIndex.from_arrays([se_df.columns.get_level_values(0), pd.to_datetime(se_df.columns.get_level_values(1))])
# Compute the rolling mean squared errors 
mse_df = cpi_rolling_mse(se_df, 30)

# Compute the BIC for the GNAR models
p_list = [1, 2, 12, 13, 25]
bic_1 = select_bic_model_order(mse_df[[f"GNAR({p},1)" for p in p_list]], 30)
bic_2 = select_bic_model_order(mse_df[[f"GNAR({p},2)" for p in p_list]], 30)
bic_s = select_bic_model_order(mse_df, 30)

def extract_order(df):
    def extract_number(s):
        start = s.find('(') + 1
        end = s.find(',')
        return int(s[start:end])
    
    df = df["Model"].apply(extract_number).copy()
    return df

def extract_neighbour_stage(df):
    def extract_number(s):
        start = s.find(',') + 1
        end = s.find(')')
        return int(s[start:end])
    
    df = df["Model"].apply(extract_number).copy()
    return df

fig,axes = plt.subplots(2, 2, figsize=(14, 5.7))

axes[0, 0].plot(dates, bic_p)
axes[0, 0].set_ylim(0.5, 13.5)
axes[0, 0].set_yticks(range(1,14,2))
axes[0, 0].set_title(r"AR($p$)")
axes[0, 0].set_xticklabels([])
axes[0, 0].set_ylabel("Model Order")

axes[0, 1].plot(dates, extract_order(bic_1))
axes[0, 1].set_ylim(0.5, 13.5)
axes[0, 1].set_yticks(range(1,14,2))
axes[0, 1].set_yticklabels([])
axes[0, 1].set_title(r"GNAR($p, \boldsymbol{1}$)")
axes[0, 1].set_xticklabels([])

axes[1, 0].plot(dates, extract_order(bic_2))
axes[1, 0].set_ylim(0.5, 13.5)
axes[1, 0].set_yticks(range(1,14,2))
axes[1, 0].set_title(r"GNAR($p, \boldsymbol{2}$)")
axes[1, 0].set_ylabel("Model Order")

axes[1, 1].plot(dates, extract_order(bic_s))
axes[1, 1].set_ylim(0.5, 13.5)
axes[1, 1].set_yticks(range(1,14,2))
axes[1, 1].set_yticklabels([])
axes[1, 1].set_title(r"GNAR($p, \boldsymbol{s}$)")

ax = axes[1, 1].twinx()
ax.plot(dates, extract_neighbour_stage(bic_s), c="r", alpha=0.7)
ax.set_ylim(0.96, 2.04)
ax.set_yticks([1, 2])
ax.grid(False)
ax.set_ylabel("Neighbour Stage", rotation=-90)

plt.tight_layout()
plt.savefig("figures/figure_3.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()