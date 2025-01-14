import requests
import os
import json
import pandas as pd
import numpy as np

# Download the CPI data from the ONS website and prepare this for the article

os.makedirs("data", exist_ok=True)

def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as output_file:
        output_file.write(response.content)

# Download the data from the ONS website and load this into pandas
url = "https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindices/current/mm23.csv"
filename = "data/cpi_data.csv"
download_file(url, filename)
cpi_data = pd.read_csv("data/cpi_data.csv")
cpi_data = cpi_data.rename(columns={"Title" : "Date"})

# CPI time series
cpi_index_cols = cpi_data.columns[cpi_data.columns.str.contains("CPI INDEX ")]
cpi_index_sorted_cols = ["Date"]
# Sort the columns
for i in range(4):
    s = sorted([name for name in cpi_index_cols if name[:21].count('.') == i])
    cpi_index_sorted_cols = cpi_index_sorted_cols + s
cpi_index_data = cpi_data[cpi_index_sorted_cols]
# The dataframe contains observations of a yearly, quaterly and monthly nature.
# The first index we are interested in is 1947 JUN. All data prior to this index is not useful.
# Hence we drop the data prior to the index of this date.
starting_index = cpi_data.index[(cpi_index_data.iloc[:, 0] == "1947 JUN")][0]
cpi_index_data = cpi_index_data.loc[starting_index:]
# Drop the rows with all NaN values
cpi_index_data = cpi_index_data.dropna(subset=cpi_index_data.columns[1:], how="all")
# Convert the date column to datetime and set it as the index
cpi_index_data["Date"] = pd.to_datetime(cpi_index_data["Date"], format="%Y %b")
cpi_index_data = cpi_index_data.set_index("Date")
# Rename the columns, dropping the first 10 and last 9 characters
cpi_index_data.columns = cpi_index_data.columns.map(lambda x: x[10:-9])
cpi_index_data.columns = cpi_index_data.columns.str.replace(":", " ").str.replace(",", ", ").str.replace("  ", " ").str.replace("  ", " ")

# %%
# Create a dictionary for the columns
split_columns = cpi_index_data.columns.str.split(' ', n=1, expand=True)
col_names = []
cpi_dict = {}
for a,b in split_columns:
    col_names.append(a)
    cpi_dict[a] = b.lower()
cpi_index_data.columns = col_names

# Save cpi_index_data and cpi_weights as CSV files
cpi_index_data.to_csv("data/cpi_monthly_data.csv")

# Save cpi_dict as a JSON file
with open("data/cpi_dict.json", "w") as f:
    json.dump(cpi_dict, f)