import pandas as pd
import numpy as np
import os

# Load the Bank of England forecasts
data_path = os.path.join(os.path.dirname(__file__), "data", "BoE_forecasts.csv")
bank_of_england = pd.read_csv(data_path, index_col=0)
bank_of_england.index = pd.to_datetime(bank_of_england.index)
bank_of_england.columns = bank_of_england.columns.astype(int)

# Load the monthly CPI data and calculate the inflation rate
data_path = os.path.join(os.path.dirname(__file__), "data", "cpi_monthly_data.csv")
cpi_index_data = pd.read_csv(data_path, index_col=0)
cpi_index = cpi_index_data[["00"]]
inflation_rate = cpi_index.pct_change(12).dropna(how="all") * 100
inflation_rate.columns = ["Inflation Rate"]
inflation_rate.index = pd.to_datetime(inflation_rate.index)


def rmse_vs_bank(preds_df):
    """
    Compute the RMSE of the Bank of England and RaGNAR models for each forecast horizon on matching dates.

    Params:
        preds_df: pd.DataFrame. DataFrame containing RaGNAR predictions. Multi-indexed by model and forecast horizon.
    
    Returns:
        pd.DataFrame. DataFrame containing the RMSE values for each forecast horizon and model.
    """
    # Create a DataFrame to store the RMSE values
    models = preds_df.columns.levels[0].to_list()
    cols = ["BoE"] + models
    rmse_df = pd.DataFrame(columns=cols, index=np.arange(1, 7))

    # Last date from which the true one-step ahead inflation rates are available
    boe = bank_of_england.loc[:"2024-11-01"]
    for i in range(1, 7):
        boe_i = boe[[i]].dropna()
        # Shift the inflation rate by i months to align with the forecast horizon
        inf_df = inflation_rate.shift(-i).loc[boe_i.index].dropna()
        dates = inf_df.index
        # Convert the DataFrames to numpy arrays
        boe_i = boe_i.loc[dates].to_numpy()
        preds = preds_df.loc[dates, pd.IndexSlice[:, i]].to_numpy()
        inf = inf_df.to_numpy()
        # Compute the RMSE for each model
        rmse_df.loc[i, "BoE"] = np.sqrt(np.mean((boe_i - inf) ** 2))
        rmse_df.loc[i, models] = np.sqrt(np.mean((preds - inf) ** 2, axis=0))
    return rmse_df 


def mape_vs_bank(preds_df, eps=1):
    """
    Compute the MAPE of the Bank of England and RaGNAR models for each forecast horizon on matching dates.

    Params:
        preds_df: pd.DataFrame. DataFrame containing RaGNAR predictions. Multi-indexed by model and forecast horizon.
        eps: float. Small value to avoid division by zero.
    
    Returns:
        pd.DataFrame. DataFrame containing the MAPE values for each forecast horizon and model.
    """
    # Create a DataFrame to store the MAPE values
    models = preds_df.columns.levels[0].to_list()
    cols = ["BoE"] + models
    mape_df = pd.DataFrame(columns=cols, index=np.arange(1, 7))

    # Last date from which the true one-step ahead inflation rates are available
    boe = bank_of_england.loc[:"2024-11-01"]
    for i in range(1, 7):
        boe_i = boe[[i]].dropna()
        # Shift the inflation rate by i months to align with the forecast horizon
        inf_df = inflation_rate.shift(-i).loc[boe_i.index].dropna()
        dates = inf_df.index
        # Convert the DataFrames to numpy arrays
        boe_i = boe_i.loc[dates].to_numpy()
        preds = preds_df.loc[dates, pd.IndexSlice[:, i]].to_numpy()
        inf = inf_df.to_numpy()
        # Compute the MAPE for each model
        mape_df.loc[i, "BoE"] = np.mean(np.abs(boe_i - inf) / (np.abs(inf) + eps) * 100)
        mape_df.loc[i, models] = np.mean(np.abs(preds - inf) / (np.abs(inf) + eps) * 100, axis=0)
    return mape_df 


def bank_scatter(preds_df, models, h, ax=None):
    """
    Create a scatter plot comparing the Bank of England forecasts to RaGNAR model forecasts for a given forecast horizon.

    Params:
        preds_df: pd.DataFrame. DataFrame containing RaGNAR predictions. Multi-indexed by model and forecast horizon.
        models: list. List of model names.
        h: int. Forecast horizon.
        ax: matplotlib.axes.Axes. Axes object to plot the scatter plot.
    """
    if ax is None:
        fig,ax = plt.subplots()
    lines = []
    for model in models:
        # Bank of England forecasts
        b = boe.loc[:"2024-11-01"][h].dropna()
        # RaGNAR model forecasts
        g = preds_df.loc[b.index, (model, h)].to_numpy().flatten()
        # True inflation rates
        i = inflation_rate.shift(-h).loc[b.index].to_numpy().flatten()
        b = b.to_numpy().flatten()
        # Percentage errors
        pe_b = (b - i) / i * 100
        pe_g = (g - i) / i * 100
        line = ax.scatter(pe_b, pe_g, label=model, s=25)
        lines.append(line)

    # Make axes equal
    ax.set_aspect('equal', adjustable='box')
    
    # Plot y=x line
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    ax.fill_betweenx(np.linspace(min_val, max_val, 100), -10, 10, color='gray', alpha=0.2)
    ax.fill_between(np.linspace(min_val, max_val, 100), -10, 10, color='gray', alpha=0.2)

    ticks = np.arange(np.floor(min_val / 10) * 10, np.ceil(max_val / 10) * 10 + 1, 10)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # Adjust limits to ensure the line spans the entire plot
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    # Apply percentage formatter to x and y axes
    ax.xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    ax.grid(True)
    ax.set_xlabel("Bank of England Percentage Error", fontsize=12)
    ax.set_ylabel("Model Percentage Error", fontsize=12)

    return lines

# Define percentage formatter
def percentage_formatter(x, pos):
    return f'{x:.0f}%'


