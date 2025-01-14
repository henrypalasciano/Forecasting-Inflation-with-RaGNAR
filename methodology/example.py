# %%
import numpy as np
import pandas as pd
from scoring import *
from random_graphs import *
from forecasting import *
# %%
# Generate some random data
data = np.random.randn(120, 5)
columns = ["00", "01", "02", "03", "04"]
index = pd.date_range("2010-01-01", "2019-12-31", freq="MS")
df = pd.DataFrame(data, index=index, columns=columns)
# Generate some adjacency matrices and compute the neighbour set matrices
adj_mats = generate_erdos_graphs(10, 5, 0.25)
ns_mats = compute_ns_mats(adj_mats, 2)
# %%
# Compute the 1-step ahead forecast errors at the first node
se_df = cpi_rolling_se(df, ns_mats, [1,2], 2, "2012-01-01", "2019-11-01", 12, 1) 
mse_df = cpi_rolling_mse(se_df, 3)
# %%
forecasts = forecast_networks(mse_df, df, adj_mats, [1,2], 2, model_type="standard", n_train=10, n_test=1, start_date="2013-03-01", end_date="2013-12-01", h=12, n_best=2)
avg_forecasts = compute_avg_preds(forecasts, 2)
avg_forecasts
# %%
mse = compute_mse_df(avg_forecasts)
mse