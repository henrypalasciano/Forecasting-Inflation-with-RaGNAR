import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.ticker import PercentFormatter
sns.set_style("white")

data_path = os.path.join(os.path.dirname(__file__), "data", "cpi_monthly_data.csv")

cpi_index = pd.read_csv(data_path, index_col=0)[["00"]]
inflation_rate = cpi_index.pct_change(12).dropna(how="all") * 100
inflation_rate.columns = ["Inflation Rate"]
inflation_rate.index = pd.to_datetime(inflation_rate.index)

def plot_predictions(inf_preds_df, steps_ahead, models, start_date, end_date, ax=None):
    preds_df = inf_preds_df.xs(steps_ahead, level=1, axis=1)[models].copy()
    preds_df = reindex_predictions(preds_df, steps_ahead)
    inf_rate = inflation_rate.loc[start_date:end_date]
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,4))
    lines = []
    line = ax.plot(inf_rate, c="k", linewidth=2)
    lines.append(line[0])
    preds_df = preds_df.loc[start_date:end_date]
    for column in preds_df.columns:
        line = ax.plot(preds_df[[column]], label=column)
        lines.append(line[0])
    plt.ylabel("Inflation Rate (%)")
    plt.title("Steps Ahead: " + str(steps_ahead))
    return lines

def reindex_predictions(inf_preds_df, steps_ahead):
    start = inflation_rate.index.get_loc(inf_preds_df.index[0]) + steps_ahead
    end = inflation_rate.index.get_loc(inf_preds_df.index[-1]) + steps_ahead + 1
    new_idx = inflation_rate.index[start:end]
    inf_preds_df = inf_preds_df.iloc[:len(new_idx)]
    inf_preds_df.index = new_idx
    return inf_preds_df


def create_grid_of_plots(inf_preds_list, steps_ahead, start_date, end_date, models, save=False, name="comparison.pdf"):
    num_rows = len(steps_ahead)
    num_cols = 3
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) + pd.Timedelta(days=90)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    y_ticks = np.arange(0, 15, 2)
    y_lim = (-2.5, 15.5)
    y_labels = [f"{step}-Month Horizon" for step in steps_ahead]

    titles = [r"Global-$\alpha$ GNAR Forecasts", r"Standard GNAR Forecasts", r"Local-$\alpha\beta$ GNAR Forecasts"]

    for i,ax_i in enumerate(axs):
        for j,ax in enumerate(ax_i):
            lines = plot_predictions(inf_preds_list[j], steps_ahead[i], models, start_date, end_date, ax)
            ax.set_ylim(y_lim)
            ax.set_title("")
            if i == num_rows - 1:
                ax.tick_params(axis='both', which='both', bottom=True, labelsize=12)
                ax.set_xlabel("")
            elif i == 0:
                ax.set_title(titles[j], fontsize=12)
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(y_labels[i], fontsize=12)
                ax.set_yticks(y_ticks)
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
                ax.tick_params(axis='both', which='both', left=True, labelsize=12)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")
            ax.set_xlim((start, end))
            ax.grid(True)
            ax.set_yticks([-2,0,2,4,6,8,10,12,14])
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.075)
    lines[0].set_label("Inflation Rate")
    fig.legend(handles=lines, loc='upper center', ncol=6, bbox_to_anchor=(0.512, 1.03), fontsize=12)
    if save:
        plt.savefig(name, format=name[-3:], dpi=300, bbox_inches='tight')
    plt.show()
    return None