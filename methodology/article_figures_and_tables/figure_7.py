import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
sns.set(style="whitegrid")

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

# Load the data and calculate the inflation rate
cpi_monthly_data = pd.read_csv("../data/cpi_monthly_data.csv", index_col=0)
cpi_monthly_data.index = pd.to_datetime(cpi_monthly_data.index)
cpi_data_pct_12 = cpi_monthly_data.pct_change(12).dropna(how="all").bfill() * 100
lf = cpi_data_pct_12[["04.5.3"]].copy()
of = cpi_data_pct_12[["01.1.5"]].copy()
fl = cpi_data_pct_12[["07.2.2"]].copy()

# Compute the inflation rate
cpi = cpi_monthly_data[["00"]]
inflation_rate = cpi.pct_change(12).dropna(how="all") * 100
inflation_rate.columns = ["Inflation Rate"]

fig,ax = plt.subplots(figsize=(15,3.5))

ax.plot(lf.sub(lf.mean()).div(lf.std()), label="Liquid Fuels")
ax.plot(of.sub(of.mean()).div(of.std()), label="Oils & Fats")
ax.plot(fl.sub(fl.mean()).div(fl.std()), label="Fuels & Lubricants")
ax.plot(inflation_rate.sub(inflation_rate.mean()).div(inflation_rate.std()), c="k", label="Inflation Rate")

ax.set_xlim(pd.to_datetime(["2010-01-01", "2024-12-01"]))
plt.legend(fontsize=12, ncol=2)

ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

plt.tight_layout()
plt.savefig("figures/figure_7.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()