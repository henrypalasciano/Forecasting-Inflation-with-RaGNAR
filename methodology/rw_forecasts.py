import pandas as pd
import numpy as np

def rolling_rw_forecast(df, q, h, start, end):
    """
    Perform rolling random walk forecasting on a time series.

    Params:
        df (pd.DataFrame): time series dataframe with index as dates and a single column.
        q (int): window size for the random walk.
        h (int): forecast horizon.
        start (str or pd.Timestamp): start date for rolling forecasting.
        end (str or pd.Timestamp): end date for rolling forecasting.

    Returns:
        pd.DataFrame: dataframe with index as the forecast start date and columns as forecast horizons.
    """
    # Construct an empty dataframe to store the random walk forecasts
    columns = pd.MultiIndex.from_product([[f"RW({q})"], range(1, h + 1)])
    index = df.loc[start : end].index
    rw_preds_df = pd.DataFrame(columns=columns, index=index, dtype=float)
    # Perform rolling random walk forecasting
    preds = df.rolling(q).mean()
    rw_preds_df.loc[:, :] = preds.loc[start : end].to_numpy().reshape(-1,1)
    return rw_preds_df