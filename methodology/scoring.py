import numpy as np
import pandas as pd

def format_data(data, p, s):
    """
    Format the data to fit GNAR models on a rolling basis

    Params:
        data: np.array. Time series and neighbour sums. Shape (m, n, max(s_vec) + 1)
        p: int. Number of lags
        s: int. Maximum stage of neighbour dependence + 1
    
    Returns:
        X: np.array. Design matrix. Shape (m - p, n, p + sum(s))
        y: np.array. Target matrix. Shape (m - p, n)
    """
    m, _ = np.shape(data)
    X = np.zeros([m - p, p * s])
    y = data[p:, 0]
    for i in range(p):
        X[:, i::p] = data[p - i - 1 : m - i - 1]
    return X, y


def cpi_rolling_se(ts_df: pd.DataFrame, A_tensor: np.array, p_list: list, s_max: int, start_date: str, 
                   end_date: str, n_train: int, n_shift: int) -> pd.DataFrame:
    """
    Compute the CPI node squared errors for a range of GNAR models, adjacency matrices and periods of data in a rolling basis

    Params:
        ts_df: pd.DataFrame. Time series data. Shape (m, n)
        A_tensor: np.array. Adjacency tensor. Shape (k, n, n, s_max)
        p_list: list. List of lags to consider
        s_max: int. Maximum stage of neighbour dependence
        start_date: str. Start date for the rolling window
        end_date: str. End date for the rolling window
        n_train: int. Number of training observations to use
        n_shift: int. Number of observations before retraining all networks again

    Returns:
        mse_test_df: pd.DataFrame. Network scores at each time step
    """
    # Maximum lag and number of observations
    p_max = np.max(p_list)

    # Extract the time series data and neighbour sums at the corresponding dates
    start = ts_df.index.get_loc(start_date) - (n_train + p_max)
    end = ts_df.index.get_loc(end_date) + 1
    ts_all = ts_df.to_numpy()[start:end]
    ts = ts_all[:, 0].reshape(-1, 1)
    ns = ts_all @ np.transpose(A_tensor[:, :, :, 0], (0, 2, 1))
    # Data shapes and number of models
    m, _ = ts.shape
    k = len(A_tensor)

    # Create a dataframe to hold the mean squared errors on the validation set
    model_names = [f"GNAR({p},{s})" for p in p_list for s in range(1, s_max + 1)]
    idx_dates = ts_df.loc[start_date:end_date].index
    dates = idx_dates[::n_shift]
    columns = pd.MultiIndex.from_product([model_names, idx_dates])
    # Create a multi-index for the columns
    se_df = pd.DataFrame(columns=columns, index=np.arange(0, k), dtype=float)
    
    # Array to store the cpi time series and it's neighbour sums
    data = np.zeros([m, 1 + s_max])
    data[:, 0] = ts.reshape(-1)
    # Array to store the mean squared errors for each iteration
    n_dates = len(dates)
    se_array = np.zeros([len(model_names) * len(idx_dates)])
    for i in range(k):
        # Fill the neighbour sums for the current iterations
        data[:, 1:] = ns[i]
        pos = 0
        # Fit the GNAR models for each lag and stage of neighbour dependence
        for p in p_list:
            # Format the data
            X, y = format_data(data[p_max-p:], p, s_max+1)
            for s in range(1, s_max+1):
                for n in range(n_dates):
                    # Fit the model for each period
                    se = compute_se(X[n * n_shift : (n + 1) * n_shift + n_train, : p * (s + 1)], y[n * n_shift : (n + 1) * n_shift + n_train], n_train)
                    se_array[pos * n_shift : (pos + 1) * n_shift] = se
                    pos += 1
        se_df.loc[i] = se_array
    return se_df

def compute_se(X, y, n_train):
    """
    Compute the squared errors for a GNAR model

    Params:
        X: np.array. Design matrix. Shape (m, n, p + sum(s))
        y: np.array. Target matrix. Shape (m, n)
        n_train: int. Number of training observations to use
    
    Returns:
        mse_train: np.array. Array of in-sample mean squared errors
        mse_test: np.array. Array of out-of-sample mean squared errors
    """
    m, _ = np.shape(X)
    X = np.hstack([np.ones([m, 1]), X])
    X_sum = np.sum(np.abs(X), axis=0)
    X = X[:, X_sum > 0]
    beta = np.linalg.lstsq(X[:n_train], y[:n_train], rcond=None)[0]
    se = (y[n_train:] - X[n_train:] @ beta) ** 2
    return se

def cpi_rolling_mse(se_df, n_val):
    models = se_df.columns.get_level_values(0).unique().to_list()
    mse_df = pd.DataFrame(columns=se_df.columns, index=se_df.index, dtype=float)
    for model in models:
        mse = se_df[model].T.rolling(window=n_val).mean().T
        mse_df.loc[:, model] = mse.to_numpy()
    return mse_df.dropna(how="all", axis=1)

