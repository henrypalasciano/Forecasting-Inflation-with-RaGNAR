import numpy as np
import pandas as pd
import sys
import os

output = "tables/table_10.txt"

with open(output, "w") as f:
    pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from forecasting import compute_rmse_df

# To load the results use the load_results function defined now
def load_results(path):
    results = pd.read_csv(path, index_col=0, header=[0, 1])
    results.columns = pd.MultiIndex.from_tuples([
        (level_0, int(level_1)) for level_0, level_1 in results.columns
    ])
    results.index = pd.to_datetime(results.index)
    return results

ar = load_results("../../results/benchmarks/fixed_order_ar_forecasts.csv")
rmses = compute_rmse_df(ar)

with open(output, "a") as f:
    print(rmses.T[[1,3,6,9,12]].to_latex(float_format="%.2f"), file=f)