# %%
import numpy as np
import pandas as pd
from random_graphs import generate_erdos_graphs, compute_ns_mats
from forecasting import forecast_networks, compute_avg_preds, compute_rmse_df, avgnar
from scoring import cpi_rolling_se, cpi_rolling_mse
# %%
# Generate 10000 random networks and compute the neighbour_set matrices
adj_mats = generate_erdos_graphs(10000, 112, 0.03)
ns_mats = compute_ns_mats(adj_mats, 2)
# %%
# Load the data
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
# Define the model lags
p_list = [1, 2, 13, 25]
# %%
# Compute the rolling standard errors and mean squared errors used to rank the networks
se_df = cpi_rolling_se(cpi_data_pct_12, ns_mats, p_list, 2, start_date="2007-07-01", end_date="2024-11-01", n_train=150, n_shift=1) 
mse_df = cpi_rolling_mse(se_df, 30)
# %%
# Compute forecasts from the top 5 networks using the global model class
glo_preds = forecast_networks(mse_df, cpi_data_pct_12, adj_mats, p_list, 2, model_type="global", n_train=150, n_test=1, start_date="2009-12-01", end_date="2024-12-01", h=12, n_best=5)
# %%
# Compute forecasts from the top 5 networks using the standard model class
std_preds = forecast_networks(mse_df, cpi_data_pct_12, adj_mats, p_list, 2, model_type="standard", n_train=150, n_test=1, start_date="2009-12-01", end_date="2024-12-01", h=12, n_best=5)
# %%
# Compute forecasts from the top 5 networks using the local model class
loc_preds = forecast_networks(mse_df, cpi_data_pct_12, adj_mats, p_list, 2, model_type="local", n_train=150, n_test=1, start_date="2009-12-01", end_date="2024-12-01", h=12, n_best=5)
# %%
# Average the forecasts from the top 5 networks
glo_avg_5 = compute_avg_preds(glo_preds, 5)
std_avg_5 = compute_avg_preds(std_preds, 5)
loc_avg_5 = compute_avg_preds(loc_preds, 5)
# %%
# Compute the RMSE for the average forecasts
glo_rmse = compute_rmse_df(glo_avg_5)
std_rmse = compute_rmse_df(std_avg_5)
loc_rmse = compute_rmse_df(loc_avg_5)
# %%
# Average the forecasts across different models (in this case GNAR(1,1), GNAR(13,1) and GNAR(25,1))
glo_avgnar = avgnar(glo_avg_5, [1,13,25], [1])
std_avgnar = avgnar(std_avg_5, [1,13,25], [1])
loc_avgnar = avgnar(loc_avg_5, [1,13,25], [1])
# %%
