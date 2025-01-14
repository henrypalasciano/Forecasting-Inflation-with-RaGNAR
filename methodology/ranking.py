import numpy as np 
import pandas as pd

def get_n_smallest(df, n):
    # Create a DataFrame to hold the indices of the smallest entries
    smallest_indices = pd.DataFrame(index=np.arange(n), columns=df.columns, dtype=int)
    # Get the smallest entries and their indices for each column
    for col in df.columns:
        smallest_indices[col] = df[col].nsmallest(n).index
    smallest_indices = smallest_indices.sort_values(by=list(df.columns))
    return smallest_indices.sort_index()