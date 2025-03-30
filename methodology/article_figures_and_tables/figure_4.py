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
from plotting import create_grid_of_plots

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
avar = load_results("../../results/benchmarks/avar_forecasts.csv")

models = ["AvGNAR({1,13,25},{1})", "AvGNAR({1,13,25},{2})", "AvGNAR({1,13,25},{1,2})"]

glo = pd.concat([avgnar_glo_5[models], avar], axis=1)
std = pd.concat([avgnar_std_5[models], avar], axis=1)
loc = pd.concat([avgnar_loc_5[models], avar], axis=1)

models = models + ["AvAR({2,13,25})"]

create_grid_of_plots([glo, std, loc], [3, 6, 9, 12], "2018-06-01", "2024-11-01", models, save=True, name="figures/figure_4.pdf")