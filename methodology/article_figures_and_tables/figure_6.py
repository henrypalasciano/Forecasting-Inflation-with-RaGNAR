import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import sys
import os
import json
sns.set(style="whitegrid")

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scoring import cpi_rolling_mse
from forecasting import get_n_smallest
from random_graphs import compute_ns_mats

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

# Load the JSON dictionary
with open("../data/cpi_dict.json", "r") as f:
    cpi_dict = json.load(f)

# Load the data
cpi_monthly_data = pd.read_csv("../data/cpi_monthly_data.csv", index_col=0)
cpi_monthly_data.index = pd.to_datetime(cpi_monthly_data.index)
to_drop = ["04.4", "04.4.1", "04.4.3", "04.5", "04.5.1", "04.5.2", 
           "08.1", "09.2.1/2/3", "10", "10.1/2/5", "10.4", "12.6.2"]
cpi_monthly_data = cpi_monthly_data.iloc[:,:124].drop(columns=to_drop)
cpi_data_pct_12 = cpi_monthly_data.pct_change(12).dropna(how="all").bfill() * 100

def get_neighbours(A, s, node=0):
    A_tensor = compute_ns_mats(A[None, :, :], s)[0]
    names = []
    for i in range(s):
        n_n = cpi_data_pct_12.columns[A_tensor[i, node] > 0].to_list()
        names_i = []
        for i in n_n:
            names_i.append(i)
        names.append(names_i)
    return names

def count_neighbours(mse_df, adj_mats, s, n_best, smallest=True):
    ts_names = cpi_data_pct_12.columns
    dates = mse_df.columns
    counts_df = pd.DataFrame(0, columns=ts_names, index=dates)
    if smallest:
        best_models = get_n_smallest(mse_df, n_best)
    else:
        best_models = get_n_largest(mse_df, n_best)
    for date in dates:
        current_models = best_models[date].to_list()
        for i,model in enumerate(current_models):
            names = get_neighbours(adj_mats[model], s)
            for j in range(s):
                for name in names[j]:
                    counts_df.loc[date, name] += 1
    return counts_df

c5 = count_neighbours(mse_df['GNAR(2,1)'], adj_mats, 1, 5).drop(columns=['00'])
c100 = count_neighbours(mse_df['GNAR(2,1)'], adj_mats, 1, 100).drop(columns=['00'])
c5.columns = c5.columns.map(cpi_dict).str.capitalize()
c100.columns = c100.columns.map(cpi_dict).str.capitalize()
c5.index = pd.to_datetime(c5.index)
c100.index = pd.to_datetime(c100.index)

fig,ax = plt.subplots(2, 1, figsize=(15, 7))
ax = ax.flatten()

c20 = count_neighbours(mse_df['GNAR(2,1)'], adj_mats, 1, 20).drop(columns=['00'])
c20.columns = c20.columns.map(cpi_dict).str.capitalize()
c20.index = pd.to_datetime(c20.index)
components = c20.rolling(24).sum().max().nlargest(10).index
labels = components.to_list()

top_5 = c5[components].div(5).rolling(6).mean().loc["2010":].copy()
top_5.columns = labels
top_100 = c100[components].div(100).rolling(6).mean().loc["2010":]
top_100.columns=labels

lines = []
for label in labels:
    line = ax[0].plot(top_5[[label]], label=label)
    lines.append(line[0])
    ax[1].plot(top_100[[label]], label=label)
ax[0].set_title("Top 5 Networks")
ax[1].set_title("Top 100 Networks")

ax[0].set_ylim((-0.05, 1.05))
ax[0].yaxis.set_major_formatter(PercentFormatter(1))

ax[1].set_ylim((-0.05, 1.05))
ax[0].set_xlabel('')
ax[1].set_yticklabels([])

ax[0].set_xticklabels([])

for a in ax:
    a.grid(True)
    a.set_xlim((pd.to_datetime('2010-01-01'), pd.to_datetime('2024-12-01')))
    a.set_ylim((-0.05, 1.05))
    a.yaxis.set_major_formatter(PercentFormatter(1))
    a.set_ylabel('Percentage of Networks', fontsize=12)

plt.subplots_adjust(wspace=0.025, hspace=0.5)

fig.legend(handles=lines, loc='upper center', ncol=5, bbox_to_anchor=(0.512, 0.5725), fontsize=12)

plt.savefig("figures/figure_6.pdf", format="pdf", dpi=300)
plt.show()