import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import ar_select_order

def rolling_ar_forecast(df, p, n, h, start, end):
    """
    Perform rolling AR(p) forecasting on a time series.
    
    Params:
        df (pd.DataFrame): time series dataframe with index as dates and a single column.
        p (int): autoregressive order.
        n (int): number of training observations.
        h (int): forecast horizon.
        start (str or pd.Timestamp): start date for rolling forecasting.
        end (str or pd.Timestamp): end date for rolling forecasting.
    
    Returns:
        pd.DataFrame: dataframe with index as the forecast start date and columns as forecast horizons.
    """
    forecasts = []
    forecast_dates = []
    
    date_range = df.loc[start:end].index
    
    for date in date_range:
        # Training data consists of the last n observations before the forecast date
        train_data = df.loc[:date].iloc[-n:].to_numpy()
        
        # Fit AR(p) model
        model = sm.tsa.AutoReg(train_data, lags=p).fit()
        forecast = model.predict(start=n, end=n + h - 1)
        
        forecasts.append(forecast)
        forecast_dates.append(date)
    
    # Convert forecasts into DataFrame
    columns = pd.MultiIndex.from_product([[f"AR({p})"], range(1, h + 1)])
    forecast_df = pd.DataFrame(forecasts, index=date_range, columns=columns)
    
    return forecast_df

def rolling_bic_ar_forecast(df, n, h, start, end):
    """
    Perform rolling AR(p) forecasting on a time series.
    
    Params:
        df (pd.DataFrame): time series dataframe with index as dates and a single column.
        p (int): autoregressive order.
        n (int): number of training observations.
        h (int): forecast horizon.
        start (str or pd.Timestamp): start date for rolling forecasting.
        end (str or pd.Timestamp): end date for rolling forecasting.
    
    Returns:
        pd.DataFrame: dataframe with index as the forecast start date and columns as forecast horizons.
    """
    forecasts = []
    forecast_dates = []
    
    date_range = df.loc[start:end].index
    bic_p = []
    
    for date in date_range:
        # Training data consists of the last n observations before the forecast date
        train_data = df.loc[:date].iloc[-n:].to_numpy()
        p = ar_select_order(train_data, 24).ar_lags[-1]
        bic_p.append(p)
        
        # Fit AR(p) model
        model = sm.tsa.AutoReg(train_data, lags=p).fit()
        forecast = model.predict(start=n, end=n + h - 1)
        
        forecasts.append(forecast)
        forecast_dates.append(date)
    
    # Convert forecasts into DataFrame
    columns = pd.MultiIndex.from_product([["AR(p)"], range(1, h + 1)])
    forecast_df = pd.DataFrame(forecasts, index=date_range, columns=columns)
    
    return forecast_df, bic_p