import numpy as np
import pandas as pd


def bic_gnar(rmse_df, n_val):
    models = rmse_df.columns.levels[0]
    bic_df = np.log(rmse_df.copy() ** 2)
    for model in models:
        k = int(model[-4]) * int(model[-2]) + 1
        bic_df[model] += k * np.log(n_val) / n_val
    bic_df = bic_df.stack(level=0).groupby(level=1).mean()
    return bic_df.loc[models]

def bic_gnar_forecasts(forecast_df, models, s):
    columns = pd.MultiIndex.from_product([[f"GNAR(p,{s})"], range(1, 13)])
    gnar_bic_forecast = pd.DataFrame(index=models.index, columns=columns)
    for date in models.index:
        gnar_bic_forecast.loc[date] = forecast_df.loc[date, models.loc[date, "Model"]].to_numpy().reshape(-1)
    return gnar_bic_forecast

def get_model_type(bic_df):
    models = bic_df.idxmin().to_frame()
    models.columns = ["Model"]
    return models

def extract_order(df):
    def extract_number(s):
        start = s.find('(') + 1
        end = s.find(',')
        return int(s[start:end])
    
    df = df["Model"].apply(extract_number).copy()
    return df

def extract_neighbour_stage(df):
    def extract_number(s):
        start = s.find(',') + 1
        end = s.find(')')
        return int(s[start:end])
    
    df = df["Model"].apply(extract_number).copy()
    return df