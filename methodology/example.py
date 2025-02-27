# %%
import numpy as np
import pandas as pd
from scoring import *
from random_graphs import *
from forecasting import *
# %%
adj_mats = generate_erdos_graphs(10000, 112, 0.03)
ns_mats = compute_ns_mats(adj_mats, 2)
# %%
cpi_monthly_data = pd.read_csv("data/cpi_monthly_data.csv", index_col=0)
cpi_monthly_data.index = pd.to_datetime(cpi_monthly_data.index)
to_drop = ["04.4", "04.4.1", "04.4.3", "04.5", "04.5.1", "04.5.2", 
           "08.1", "09.2.1/2/3", "10", "10.1/2/5", "10.4", "12.6.2"]
cpi_monthly_data = cpi_monthly_data.iloc[:,:124].drop(columns=to_drop)
cpi_data_pct_12 = cpi_monthly_data.pct_change(12).dropna(how="all").bfill() * 100

# Compute the inflation rate
cpi = cpi_monthly_data[["00"]]
inflation_rate = cpi.pct_change(12).dropna(how="all") * 100
inflation_rate.columns = ["Inflation Rate"]
# %%
p_list = [1, 2, 12, 13, 25]
# %%
se_df = cpi_rolling_se(cpi_data_pct_12, ns_mats, p_list, 2, start_date="2007-07-01", end_date="2024-11-01", n_train=150, n_shift=1) 
mse_df = cpi_rolling_mse(se_df, 30)
# %%
glo_preds = forecast_networks(mse_df, cpi_data_pct_12, adj_mats, p_list, 2, model_type="global", n_train=150, n_test=1, start_date="2009-12-01", end_date="2024-12-01", h=12, n_best=5)
# %%
std_preds = forecast_networks(mse_df, cpi_data_pct_12, adj_mats, p_list, 2, model_type="standard", n_train=150, n_test=1, start_date="2009-12-01", end_date="2024-12-01", h=12, n_best=5)
# %%
loc_preds = forecast_networks(mse_df, cpi_data_pct_12, adj_mats, p_list, 2, model_type="local", n_train=150, n_test=1, start_date="2009-12-01", end_date="2024-12-01", h=12, n_best=5)
# %%
glo_avg_5 = compute_avg_preds(glo_preds, 5)
std_avg_5 = compute_avg_preds(std_preds, 5)
loc_avg_5 = compute_avg_preds(loc_preds, 5)
# %%
glo_rmse = compute_rmse_df(glo_avg_5)
std_rmse = compute_rmse_df(std_avg_5)
loc_rmse = compute_rmse_df(loc_avg_5)
# %%
