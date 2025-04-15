import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
sns.set(style="white")

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

cpi_monthly_data = pd.read_csv("../data/cpi_monthly_data.csv", index_col=0)
cpi_monthly_data.index = pd.to_datetime(cpi_monthly_data.index)

# Compute the inflation rate
cpi = cpi_monthly_data[["00"]]
inflation_rate = cpi.pct_change(12).dropna(how="all") * 100
inflation_rate.columns = ["Inflation Rate"]

fig,axes = plt.subplots(1,3,figsize=(13.5,3))
plot_pacf(inflation_rate["1995":"2014"], lags=37, ax=axes[0])
plot_pacf(inflation_rate["2000":"2019"], lags=37, ax=axes[1])
plot_pacf(inflation_rate["2005":"2024"], lags=37, ax=axes[2])
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
axes[0].set_ylabel("Partial Autocorrelation", fontsize=12)
axes[1].set_yticklabels([])
axes[2].set_yticklabels([])
for ax in axes:
    ax.set_xticks(range(0, 38, 5))
    ax.set_xlabel("Lag", fontsize=12)
    ax.set_title("")
plt.tight_layout()
plt.savefig("figures/figure_9.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()