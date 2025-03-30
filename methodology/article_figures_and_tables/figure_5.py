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
from bank_of_england import mape_vs_bank, bank_scatter

# To load the results use the load_results function defined now
def load_results(path):
    results = pd.read_csv(path, index_col=0, header=[0, 1])
    results.columns = pd.MultiIndex.from_tuples([
        (level_0, int(level_1)) for level_0, level_1 in results.columns
    ])
    results.index = pd.to_datetime(results.index)
    return results

# Replace the paths in the load to those of a different run of RaGNAR if necessary
avgnar_glo_5 = load_results("../../results/ragnar/avgnar/global_5.csv")
avgnar_std_5 = load_results("../../results/ragnar/avgnar/standard_5.csv")
avgnar_loc_5 = load_results("../../results/ragnar/avgnar/local_5.csv")

models = ["AvGNAR({2,13,25},{1})", "AvGNAR({2,13,25},{2})", "AvGNAR({2,13,25},{1,2})"]

avgnar_glo_bank_mape = np.round(mape_vs_bank(avgnar_glo_5), 2)
avgnar_std_bank_mape = np.round(mape_vs_bank(avgnar_std_5), 2)
avgnar_loc_bank_mape = np.round(mape_vs_bank(avgnar_loc_5), 2)

fig,axes = plt.subplots(3, 3, figsize=(14,14))
lines = bank_scatter(avgnar_glo_5, models, 4, ax=axes[0,0])
lines = bank_scatter(avgnar_std_5, models, 4, ax=axes[0,1])
lines = bank_scatter(avgnar_loc_5, models, 4, ax=axes[0,2])

lines = bank_scatter(avgnar_glo_5, models, 5, ax=axes[1,0])
lines = bank_scatter(avgnar_std_5, models, 5, ax=axes[1,1])
lines = bank_scatter(avgnar_loc_5, models, 5, ax=axes[1,2])

lines = bank_scatter(avgnar_glo_5, models, 6, ax=axes[2,0])
lines = bank_scatter(avgnar_std_5, models, 6, ax=axes[2,1])
lines = bank_scatter(avgnar_loc_5, models, 6, ax=axes[2,2])

for i in [0,1,2]:
    for j in [0,1,2]:
        if i < 2:
            axes[i,j].set_xlabel("")
        if j > 0:
            axes[i,j].set_ylabel("")

axes[0,0].set_title(r"Global-$\alpha$ GNAR Forecasts")
axes[0,1].set_title(r"Standard GNAR Forecasts")
axes[0,2].set_title(r"Local-$\alpha\beta$ GNAR Forecasts")
fig.legend(handles=lines, loc='upper center', ncol=6, bbox_to_anchor=(0.512, 1.02), fontsize=12)

plt.tight_layout()
plt.savefig("figures/figure_5.pdf", format="pdf", dpi=300, bbox_inches='tight')
plt.show()
