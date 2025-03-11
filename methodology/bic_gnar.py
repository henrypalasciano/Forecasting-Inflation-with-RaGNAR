import numpy as np
import pandas as pd

def select_bic_model_order(rmse_df, n_val, n=2500):
    """
    Select the model with the lowest BIC for each forecast date, averaged across the n best networks each month.
    """
    # Get models and intialize BIC dataframe
    models = rmse_df.columns.levels[0]
    bic_df = np.log(get_n_smallest_vals(rmse_df, n).copy() ** 2)
    for model in models:
        # Compute the number of parameters for each model
        p = int(model[model.find('(') + 1 : model.find(',')])
        s = int(model[model.find(',') + 1 : model.find(')')])
        k = p * (s + 1)
        # Compute BIC
        bic_df[model] += k * np.log(n_val) / n_val
    # Stack and average BIC values (future_stack=True is used to avoid a warning)
    bic_df = bic_df.stack(level=0, future_stack=True).groupby(level=1).mean()
    model_df = bic_df.idxmin().to_frame()
    model_df.columns = ["Model"]
    return model_df

def get_n_smallest_vals(df, n):
    """
    Get the n smallest values in each column of a DataFrame.
    """
    # Create a DataFrame to hold the indices of the smallest entries
    smallest = pd.DataFrame(index=np.arange(n), columns=df.columns, dtype=int)
    # Get the smallest entries and their indices for each column
    for col in df.columns:
        smallest[col] = df[col].nsmallest(n).to_numpy()
    smallest = smallest.sort_values(by=list(df.columns))
    return smallest.sort_index()

def bic_forecasts(preds_df, model_df, name):
    """
    Select the forecasts from the best model according to the BIC for each forecast date.
    """
    # Create dataframe to hold the BIC forecasts
    columns = pd.MultiIndex.from_product([[name], range(1, 13)])
    dates = preds_df.index
    bic_preds = pd.DataFrame(index=dates, columns=columns)
    # Get the forecasts from the best model according to the BIC
    for date in dates:
        bic_preds.loc[date] = preds_df.loc[date, model_df.loc[date, "Model"]].to_numpy().reshape(-1)
    return bic_preds