import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os
import logging

from scoring import cpi_rolling_se, cpi_rolling_mse
from random_graphs import generate_erdos_graphs, compute_ns_mats
from forecasting import forecast_networks, compute_avg_preds, compute_rmse_df, compute_mape_df, avgnar
from bank_of_england import rmse_vs_bank, mape_vs_bank
from bic_gnar import select_bic_model_order, bic_forecasts

cpi_monthly_data = pd.read_csv("data/cpi_monthly_data.csv", index_col=0)
cpi_monthly_data.index = pd.to_datetime(cpi_monthly_data.index)
to_drop = ["04.4", "04.4.1", "04.4.3", "04.5", "04.5.1", "04.5.2", 
           "08.1", "09.2.1/2/3", "10", "10.1/2/5", "10.4", "12.6.2"]
cpi_monthly_data = cpi_monthly_data.iloc[:,:124].drop(columns=to_drop)
cpi_data_pct_12 = cpi_monthly_data.pct_change(12).dropna(how="all").bfill() * 100

cpi = cpi_monthly_data[["00"]]
inflation_rate = cpi.pct_change(12).dropna(how="all") * 100
inflation_rate.columns = ["Inflation Rate"]

logging.basicConfig(filename="logger.txt", level=logging.INFO, format="%(asctime)s - %(message)s", filemode="w")

# Create required directories
os.makedirs("quarterly_results", exist_ok=True)

def save_forecast_results(prefix, avg_preds_df, i, j, mtype):
    """Save RMSE, MAPE, and Bank RMSE and MAPE for a given forecast."""
    df = compute_rmse_df(avg_preds_df)
    df.to_csv(f"quarterly_results/rmse_df_{prefix}_{mtype}_{j}_{i}.csv")
    return None

logging.info("Starting Computation")

def parallel_cpi(i: int):

    # Set up logging inside the function
    logging.basicConfig(filename="logger.txt", level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info(f"Starting {i}-th run")

    p_list = [1, 2]

    success = False
    while not success:
        try:
            # Generate 10000 random networks and compute the neighbour_set matrices
            adj_mats = generate_erdos_graphs(10000, 112, 0.03)
            ns_mats = compute_ns_mats(adj_mats, 2)

            # Compute the rolling squared errors
            mse_df = cpi_rolling_se(cpi_data_pct_12, ns_mats, p_list, 2, start_date="2007-06-01", end_date="2024-11-01", n_train=150, n_shift=3)

            # Compute the rolling mean squared errors
            mse_df = cpi_rolling_mse(mse_df, 30)

            # Forecast using different models
            forecast_types = ["global", "standard", "local"]
            forecasts = {ftype: forecast_networks(mse_df, cpi_data_pct_12, adj_mats, p_list, 2, model_type=ftype, 
                                                n_train=150, n_test=3, start_date="2009-12-01", end_date="2024-12-01", h=12, n_best=5)
                        for ftype in forecast_types}
            success = True
        except Exception as e:
            logging.error(f"Error in {i}-th run: {e}. Retrying...")

    # Save forecasts
    for ftype, df in forecasts.items():
        df.to_csv(f"quarterly_results/preds_df_{ftype}_{i}.csv")

    # Get dates to construct dataframes
    dates = df.index

    bic_1 = select_bic_model_order(mse_df[[f"GNAR({p},1)" for p in p_list]], 30)
    bic_2 = select_bic_model_order(mse_df[[f"GNAR({p},2)" for p in p_list]], 30)
    bic_s = select_bic_model_order(mse_df, 30)

    for j in [1,2,5]:
        for ftype in forecast_types:
            # Average the forecasts of the top j networks
            avg_preds_df = compute_avg_preds(forecasts[ftype], j)
            save_forecast_results(ftype, avg_preds_df, i, j, "avg")
            # BIC forecasts
            bic_df = pd.concat([bic_forecasts(avg_preds_df, bic_1, "GNAR(p,1)"), bic_forecasts(avg_preds_df, bic_2, "GNAR(p,2)"), 
                                bic_forecasts(avg_preds_df, bic_s, "GNAR(p,s)")], axis=1)
            save_forecast_results(ftype, bic_df, i, j, "bic")

    logging.info(f"{i}-th run COMPLETE")

    return i

res = Parallel(n_jobs=25)(delayed(parallel_cpi)(i) for i in range(0, 100))
logging.info("Computation Complete")


