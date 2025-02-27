import pandas as pd
import numpy as np

from gnar.gnar import GNAR
from ranking import get_n_smallest

cpi_index = pd.read_csv("data/cpi_monthly_data.csv", index_col=0)[["00"]]
inflation_rate = cpi_index.pct_change(12).dropna(how="all") * 100
inflation_rate.columns = ["Inflation Rate"]
inflation_rate.index = pd.to_datetime(inflation_rate.index)
index = inflation_rate.index

def forecast_networks(results_df, ts_df, adj_mats, p_list, s_max, model_type="standard", n_train=150, 
                      n_test=1, start_date="2009-01-01", end_date=index[-2], h=12, n_best=1):
    """
    Forecast from the top n_best networks using GNAR processes.

    Params:
        results_df: pd.DataFrame. Dataframe containing the scores for each network at each time step.
        ts_df: pd.DataFrame. Time series data. Shape (m, n)
        adj_mats: np.array. Adjacency matrices. Shape (k, n, n)
        p_list: list. List of lags to consider
        s_max: int. Maximum stage of neighbour dependence
        model_type: str. GNAR model class to use
        n_train: int. Number of observations to use for training at each time-step
        n_test: int. Number of steps before retraining.
        start_date: str. Start date for the rolling window
        end_date: str. End date for the rolling window
        h: int. Forecast horizon
        n_best. Number of best performing networks to forecast from
    
    Returns:
        inflation_rate_preds_df: pd.DataFrame. Inflation rate forecasts for each best performing network at each time step
    """
    # Convert the start and end dates to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    # Best models at each time step
    best_models = get_n_smallest(results_df, n_best)
    # Construct an empty dataframe to store the inflation rate forecasts
    model_names = [f"GNAR({p},{s})" for p in p_list for s in range(1, s_max + 1)]
    dates = results_df.columns.get_level_values(1).unique().to_list()
    pred_dates = [date for date in dates if date >= start_date and date < end_date] + [end_date]
    columns = pd.MultiIndex.from_product([range(1, n_best + 1), model_names, range(1, h+1)])
    inflation_rate_preds_df = pd.DataFrame(index=ts_df.loc[pred_dates[0]:end_date].iloc[:-1].index, columns=columns, dtype=float)
    # Forecast the inflation rate for each best performing network at each time step, shifting the training window by n_test
    for i in range(0, len(pred_dates) - n_test, n_test):
        for p in p_list:
            # Train and test data
            start = pred_dates[i]
            end = pred_dates[i + n_test]
            train = ts_df.loc[:start].iloc[-(n_train + p):]
            test = ts_df.loc[:end].iloc[-(n_test + p):-1]
            # Forecast the inflation rate for each best performing network
            for s in range(1, s_max+1):
                model = f"GNAR({p},{s})"
                models = best_models[(model, start)].to_list()
                inf_preds, index = GNAR_preds(models, adj_mats, p, s, train, test, model_type, h, n_best)
                # Store the inflation rate forecasts
                inflation_rate_preds_df.loc[index, pd.IndexSlice[:, model]] = inf_preds
    return inflation_rate_preds_df

def GNAR_preds(best_models, adj_mats, p, s, train, test, model_type, h, n_best):
    """
    Computes forecasts for the inflation rate using the n_best best performing graphs.
    """
    # Number of time steps to forecast
    t = len(test) - p + 1
    inf_preds = []
    for model in best_models:
        # Fit the GNAR model and forecast inflation
        G = GNAR(A=adj_mats[model], p=p, s=np.array([s] * p), ts=train, model_type=model_type)
        inf_preds.append(G.predict(test, h)["00"].to_numpy().reshape(t, h))
    return np.hstack(inf_preds), test.index[p-1:]

def compute_avg_preds(inf_preds_df, n_best):
    """
    Compute the average inflation rate forecasts from the n_best best performing networks. (inf_preds_df is the output of forecast_networks)
    """
    inf_preds_avg = inf_preds_df[1]
    for i in range(2, n_best + 1):
        inf_preds_avg += inf_preds_df[i]
    inf_preds_avg = inf_preds_avg / n_best
    return inf_preds_avg

def avgnar(preds_df, p_list, s_list):
    """
    Average GNAR models of different orders. (preds_df is the output of compute_avg_preds)
    """
    # Number of models across which to average
    n = len(p_list) * len(s_list)
    forecast_sum = None
    # Loop over models
    for p in p_list:
        for s in s_list:
            forecasts = preds_df[f"GNAR({p},{s})"].to_numpy()
            if forecast_sum is None:
                forecast_sum = forecasts
            else:
                forecast_sum += forecasts
    # Create output dataframe
    columns = pd.MultiIndex.from_product([[f"AvGNAR({p_list},{s_list})"], preds_df.columns.levels[1]])
    output_df = pd.DataFrame(forecast_sum / n, index=preds_df.index, columns=columns)
    return output_df

def compute_mse_df(inf_preds_df, start=None, end=None):
    """
    Compute the mean squared errors for the inflation rate forecasts.
    """
    if start is not None:
        inf_preds_df = inf_preds_df.loc[start:]
    if end is not None:
        inf_preds_df = inf_preds_df.loc[:end]
    # Construct an empty dataframe to store the mean squared errors
    model_names = inf_preds_df.columns.get_level_values(0).unique().to_list()
    steps = inf_preds_df.columns.get_level_values(1).unique().to_list()
    mse_df = pd.DataFrame(index=steps, columns=model_names, dtype=float)
    dates = inf_preds_df.index
    # Compute the mean squared errors for each model at each forecast horizon
    for step in steps:
        true_inf = inflation_rate.shift(-step).loc[dates].dropna()
        mse_df.loc[step] = np.mean((inf_preds_df.loc[true_inf.index, pd.IndexSlice[:, step]].to_numpy() - true_inf.to_numpy()) ** 2, axis=0)
    return mse_df

def compute_mape_df(inf_preds_df, start=None, end=None, eps=1):
    """
    Compute the mean absolute percentage error for the inflation rate forecasts. Here eps is the correction applied to the denominator to avoid division by zero.
    """
    if start is not None:
        inf_preds_df = inf_preds_df.loc[start:]
    if end is not None:
        inf_preds_df = inf_preds_df.loc[:end]
    # Construct an empty dataframe to store the mean absolute percentage errors
    model_names = inf_preds_df.columns.get_level_values(0).unique().to_list()
    steps = inf_preds_df.columns.get_level_values(1).unique().to_list()
    mape_df = pd.DataFrame(index=steps, columns=model_names, dtype=float)
    dates = inf_preds_df.index
    # Compute the mean absolute percentage errors for each model at each forecast horizon
    for step in steps:
        true_inf = inflation_rate.shift(-step).loc[dates].dropna()
        mape_df.loc[step] = np.mean(np.abs(inf_preds_df.loc[true_inf.index, pd.IndexSlice[:, step]].to_numpy() - true_inf.to_numpy()) / (np.abs(true_inf.to_numpy()) + eps) * 100, axis=0)
    return mape_df