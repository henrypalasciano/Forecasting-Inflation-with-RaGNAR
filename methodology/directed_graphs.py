import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os

os.makedirs("dir_results", exist_ok=True)

from scoring import compute_se
from forecasting import get_n_smallest, compute_rmse_df
from gnar.gnar import GNAR
from random_graphs import generate_erdos_graphs, compute_ns_mats


def format_data_all(data, p, s):
    """
    Format the data to fit multiple GNAR models of order p and of maximum neighbour stage dependence s-1

    Params:
        data: np.array. Time series and neighbour sums. Shape (m, n, max(s_vec) + 1)
        p: int. Number of lags
        s: int. Maximum stage of neighbour dependence + 1
    
    Returns:
        X: np.array. Design matrix. Shape (m - p, n, p + sum(s))
        y: np.array. Target matrix. Shape (m - p, n)
    """
    m, n, _ = np.shape(data)
    X = np.zeros([m - p, n, p * s])
    y = data[p:, :, 0]
    for i in range(p):
        X[:, :, i::p] = data[p - i - 1 : m - i - 1, :, :]
    return X, y

def all_node_rolling_se(ts_df: pd.DataFrame, A_tensor: np.array, start_date: str, end_date: str, n_shift: int, p_list: list, s_max: int, n_train: int) -> tuple:
    """
    Compute the CPI node squared errors for a range of GNAR models, adjacency matrices and periods of data in a rolling basis

    Params:
        ts_df: pd.DataFrame. Dataframe of CPI data
        A_tensor: np.array. Array of neighbour adjacency matrices
        start_date: str.
        end_date: str.
        n_shift: int.
        p_list: list. List of lags to use
        s_max: int. Maximum stage of neighbour dependence
        n_train: int. Number of training observations to use

    Returns:
        mse_test_df: pd.DataFrame. Dataframes of in-sample and out-of-sample mean squared errors
    """
    # Maximum lag and number of observations
    p_max = np.max(p_list)

    # Extract the time series data and neighbour sums at the corresponding dates
    start = ts_df.index.get_loc(start_date) - (n_train + p_max)
    end = ts_df.index.get_loc(end_date) + 1
    ts_all = ts_df.to_numpy()[start:end]
    ns = np.transpose(ts_all @ np.transpose(A_tensor, (0, 3, 2, 1)), (0, 2, 1, 3))
    # Data shapes and number of models
    m, n_ts = ts_all.shape
    k = len(A_tensor)

    # Create a dataframe to hold the mean squared errors on the validation set
    model_names = ["GNAR(" + str(p) + "," + str(s) + ")" for p in p_list for s in range(1, s_max+1)]
    ts_names = ts_df.columns.to_list()
    idx_dates = ts_df.loc[start_date:end_date].index
    dates = idx_dates[::n_shift]
    columns = pd.MultiIndex.from_product([ts_names, model_names, idx_dates])
    # Create a multi-index for the columns
    se_df = pd.DataFrame(columns=columns, index=np.arange(0, k), dtype=float)

    # Array to store the cpi time series and it's neighbour sums
    data = np.zeros([m, n_ts, 1 + s_max])
    data[:, :, 0] = ts_all
    # Array to store the mean squared errors for each iteration
    n_dates = len(dates)
    se_array = np.zeros([n_ts, len(model_names) * len(idx_dates)])
    for i in range(k):
        # Fill the neighbour sums for the current iterations
        data[:, :, 1:] = ns[i]
        pos = 0
        # Fit the GNAR models for each lag and stage of neighbour dependence
        for p in p_list:
            # Format the data
            X, y = format_data_all(data[p_max-p:], p, s_max+1)
            for s in range(1, s_max+1):
                for n in range(n_dates):
                    for ts in range(n_ts):
                        # Fit the model for each period
                        se = compute_se(X[n * n_shift : (n + 1) * n_shift + n_train, ts, : p * (s + 1)], y[n * n_shift : (n + 1) * n_shift + n_train, ts], n_train)
                        se_array[ts, pos * n_shift : (pos + 1) * n_shift] = se
                    pos += 1
        se_df.loc[i] = se_array.flatten()
    se_df = se_df.sort_index(axis=1)
    return se_df

def all_node_rolling_mse(se_df, n_test):
    """
    Compute the rolling mean squared errors from the squared errors dataframe for all nodes
    """
    ts_names = se_df.columns.get_level_values(0).unique().to_list()
    models = se_df.columns.get_level_values(1).unique().to_list()
    # Create a dataframe to hold the mean squared errors
    mse_df = pd.DataFrame(columns=se_df.columns, index=se_df.index, dtype=float)
    se_df = se_df.sort_index()
    # Iterate over nodes and models - this insures that the rolling window does not overlap across nodes and models
    for ts in ts_names:
        for model in models:
            mse = se_df[(ts, model)].T.rolling(window=n_test).mean().T
            mse_df.loc[:, (ts, model)] = mse.to_numpy()
    return mse_df.dropna(how="all", axis=1)

def construct_best_networks(adj_mats, mse_df):
    """
    Construct directed networks using the mean squared error dataframe.
    These are stored in a dictionary of dictionaries.
    """
    # Time series indices and model names
    time_series_indices = np.arange(0, adj_mats.shape[-1])
    models = mse_df.columns.get_level_values(1).unique().to_list()
    # Forecasting dates
    dates = mse_df.columns.get_level_values(2).unique().to_list()
    # Get the index of the best network for each date, model and node
    best_nets = get_n_smallest(mse_df, 1)
    best_networks = dict()
    for model in models:
        model_dict = dict()
        # Construct an adjacency matrix for each date
        for date in dates:
            # Get the best network for the current date and model
            best_comps = best_nets.loc[:, pd.IndexSlice[:, model, date]].to_numpy().flatten()
            # Construct the adjacency matrix
            model_dict[date] = adj_mats[best_comps, time_series_indices].T
        best_networks[model] = model_dict
    return best_networks


def forecast_dir_networks(ts_df, adj_mat_dict, p_list, model_type="standard", n_train=150, 
                          n_test=1, start_date="2009-01-01", end_date="2024-11-01", h=12):
    """
    Forecast from the top n_best networks using GNAR processes.

    Params:
        ts_df: pd.DataFrame. Time series data. Shape (m, n)
        adj_mat_dict: dict. Adjacency matrix dictionary
        p_list: list. List of lags to consider
        model_type: str. GNAR model class to use
        n_train: int. Number of observations to use for training at each time-step
        n_test: int. Number of steps before retraining.
        start_date: str. Start date for the rolling window
        end_date: str. End date for the rolling window
        h: int. Forecast horizon
    
    Returns:
        inflation_rate_preds_df: pd.DataFrame. Inflation rate forecasts for each best performing network at each time step
    """
    # Convert the start and end dates to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    # Construct an empty dataframe to store the inflation rate forecasts
    model_names = [f"GNAR({p},1)" for p in p_list]
    dates = list(adj_mat_dict[f"GNAR({p_list[0]},1)"].keys())
    pred_dates = [date for date in dates if date >= start_date and date < end_date] + [end_date]
    columns = pd.MultiIndex.from_product([model_names, range(1, h+1)])
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
            model = f"GNAR({p},1)"
            A = adj_mat_dict[model][start]
            inf_preds, index = GNAR_dir_preds(A, p, 1, train, test, model_type, h)
            # Store the inflation rate forecasts
            inflation_rate_preds_df.loc[index, model] = inf_preds
    return inflation_rate_preds_df

def GNAR_dir_preds(A, p, s, train, test, model_type, h):
    """
    Computes forecasts for the inflation rate using the n_best best performing graphs.
    """
    # Number of time steps to forecast
    t = len(test) - p + 1
    inf_preds = []
    G = GNAR(A=A, p=p, s=np.array([s] * p), ts=train, model_type=model_type)
    inf_preds = G.predict(test, h)["00"].to_numpy().reshape(t, h)
    return inf_preds, test.index[p-1:]

cpi_monthly_data = pd.read_csv("data/cpi_monthly_data.csv", index_col=0)
cpi_monthly_data.index = pd.to_datetime(cpi_monthly_data.index)
to_drop = ["04.4", "04.4.1", "04.4.3", "04.5", "04.5.1", "04.5.2", 
           "08.1", "09.2.1/2/3", "10", "10.1/2/5", "10.4", "12.6.2"]
cpi_monthly_data = cpi_monthly_data.iloc[:,:124].drop(columns=to_drop)
cpi_data_pct_12 = cpi_monthly_data.pct_change(12).dropna(how="all").bfill() * 100

cpi = cpi_monthly_data[["00"]]
inflation_rate = cpi.pct_change(12).dropna(how="all") * 100
inflation_rate.columns = ["Inflation Rate"]

def parallel_dir(i):

    p_list = [1, 2]

    # Generate 10000 random networks and compute the neighbour_set matrices
    adj_mats = generate_erdos_graphs(10000, 112, 0.03)
    ns_mats = compute_ns_mats(adj_mats, 1)

    # Compute the rolling squared errors
    mse_df = all_node_rolling_se(cpi_data_pct_12, ns_mats, p_list=p_list, s_max=1, start_date="2007-07-01", end_date="2024-11-01", n_train=150, n_shift=1)
    # Compute the rolling mean squared errors
    mse_df = all_node_rolling_mse(mse_df, 30)

    adj_mat_dict = construct_best_networks(adj_mats, mse_df)

    # Forecast using different models
    forecast_types = ["global", "standard", "local"]
    forecasts = {ftype: forecast_dir_networks(cpi_data_pct_12, adj_mat_dict, p_list, model_type=ftype, n_train=150,
                                              n_test=1, start_date="2009-12-01", end_date="2024-12-01", h=12)
                 for ftype in forecast_types}

    for ftype in forecast_types:
        df = forecasts[ftype]
        df.to_csv(f"dir_results/preds_df_{ftype}_{i}.csv")
        rmse_df = compute_rmse_df(df)
        rmse_df.to_csv(f"dir_results/rmse_df_{ftype}_{i}.csv")

    return None

res = Parallel(n_jobs=10)(delayed(parallel_dir)(i) for i in range(0, 100))