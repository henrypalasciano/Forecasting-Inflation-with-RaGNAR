import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import sys
import os
sns.set(style="whitegrid")

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scoring import cpi_rolling_mse
from forecasting import get_n_smallest

# Load the squared error dataframe and the adjacency matrices
se_df_list = []
adj_mats_list = []
# Stored in 10 different files so as to not exceed the 100MB limit on GitHub - can replace this code with a single file if necessary
for i in range(1, 11):
    # Replace the path with that of the results of a different run of RaGNAR if necessary
    se_df_list.append(pd.read_csv(f"../../results/ragnar/se_df/se_df_{i}.csv", index_col=0, header=[0, 1]))
    adj_mats_list.append(np.load(f"../../results/ragnar/adj_mats/adj_mats_{i}.npy"))
se_df = pd.concat(se_df_list)
# Reconstruct the MultiIndex with the dates in the correct format
se_df.columns = pd.MultiIndex.from_arrays([se_df.columns.get_level_values(0), pd.to_datetime(se_df.columns.get_level_values(1))])
# Compute the rolling mean squared errors 
mse_df = cpi_rolling_mse(se_df, 30)
# Stack the adjacency matrices into a single numpy array
adj_mats = np.vstack(adj_mats_list)

# Load the data
cpi_monthly_data = pd.read_csv("../data/cpi_monthly_data.csv", index_col=0)
cpi_monthly_data.index = pd.to_datetime(cpi_monthly_data.index)
to_drop = ["04.4", "04.4.1", "04.4.3", "04.5", "04.5.1", "04.5.2", 
           "08.1", "09.2.1/2/3", "10", "10.1/2/5", "10.4", "12.6.2"]
cpi_monthly_data = cpi_monthly_data.iloc[:,:124].drop(columns=to_drop)
cpi_data_pct_12 = cpi_monthly_data.pct_change(12).dropna(how="all").bfill() * 100

# Compute the inflation rate
cpi = cpi_monthly_data[["00"]]
inflation_rate = cpi.pct_change(12).dropna(how="all") * 100
inflation_rate.columns = ["Inflation Rate"]

best_nets = get_n_smallest(mse_df, 1)

# Each network chosen contains the dominant item from each period.
# The networks are ordered so as to match the colours displayed in the article,
# although the time series themselves may differ.
net_A = best_nets.loc[0, ("GNAR(2,1)", "2023-01-01")]
net_B = best_nets.loc[0, ("GNAR(2,1)", "2011-01-01")]
net_C = best_nets.loc[0, ("GNAR(2,1)", "2018-01-01")]

ns1 = cpi_data_pct_12.iloc[:, adj_mats[net_A, 0] > 0].mean(axis=1).to_frame()
ns2 = cpi_data_pct_12.iloc[:, adj_mats[net_B, 0] > 0].mean(axis=1).to_frame()
ns3 = cpi_data_pct_12.iloc[:, adj_mats[net_C, 0] > 0].mean(axis=1).to_frame()

fig,ax = plt.subplots(figsize=(15, 3.5))
ax.plot(ns1, label="Network A")
ax.plot(ns2, label="Network B")
ax.plot(ns3, label="Network C")

ax.plot(inflation_rate, c="k", label="Inflation Rate")
ax.set_xlim(pd.to_datetime(["2010-01-01", "2024-12-01"]))
plt.legend()
ax.set_xlabel("Date")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

plt.tight_layout()
plt.savefig("figures/figure_8.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()