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
cpi_lvl_1 = cpi_monthly_data.iloc[:, :13]
cpi_monthly_data.index = pd.to_datetime(cpi_monthly_data.index)

# Compute the inflation rate
cpi = cpi_monthly_data[["00"]]
inflation_rate = cpi.pct_change(12).dropna(how="all") * 100
inflation_rate.columns = ["Inflation Rate"]

# Creating a 1x3 grid of subplots
cpi_lvl_1_pct = cpi_lvl_1.pct_change(12).dropna(how="all") * 100
cpi_lvl_1_pct.index = pd.to_datetime(cpi_lvl_1_pct.index)
cpi_lvl_1_pct.columns = ["Inflation Rate", "Food & Non-Alcoholic Beverages", "Alcoholic Beverages, Tobacco & Narcotics",
                         "Clothing & Footwear", "Housing, Water & Fuels", "Furniture, Household Equipment & Repairs",
                         "Health", "Transport", "Communication", "Recreation & Culture", "Education", "Hotels, Cafes & Restaurants",
                         "Miscellaneous Goods & Services"]

fig, axs = plt.subplots(2, 1, figsize=(15, 6))

# Manually plotting columns on each subplot
# Adjust the column selection as per your requirement
for i in range(6):
    axs[0].plot(cpi_lvl_1_pct.iloc[:, 1 + i].to_frame(), label=cpi_lvl_1_pct.iloc[:, 1 + i].to_frame().columns[0])
    axs[1].plot(cpi_lvl_1_pct.iloc[:, 7 + i].to_frame(), label=cpi_lvl_1_pct.iloc[:, 7 + i].to_frame().columns[0])

for ax in axs:
    ax.plot(cpi_lvl_1_pct.iloc[:, 0].to_frame(), c="k", label=cpi_lvl_1_pct.iloc[:, 0].to_frame().columns[0])
    ax.set_xlim(pd.to_datetime(["2010-01-01", "2024-12-01"]))
    ax.grid(True)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylim(-6, 28)

axs[0].set_xticklabels([])
axs[0].set_yticks([-5, 0, 5, 10, 15, 20, 25])
axs[0].legend(fontsize=12, ncol=3, loc="upper left")
axs[1].legend(fontsize=12, ncol=3, loc="upper right")

plt.tight_layout()
plt.savefig("figures/figure_1.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()