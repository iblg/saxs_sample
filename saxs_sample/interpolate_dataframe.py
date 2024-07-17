import pandas as pd
import numpy as np


def interpolate_dataframe(df1, xgrid):
    """
    This function interpolates the column 'I' in xgrid based on the 'q' column in df1.

    Args:
        df1 (pandas.DataFrame): The dataframe containing the 'q' and 'I' columns.
        xgrid (pandas.DataFrame): The dataframe containing the values for which to interpolate 'I'.

    Returns:
        pandas.DataFrame: A new dataframe with columns from xgrid and the interpolated 'I' values.
    """
    # Sort both DataFrames by the 'q' column
    df1 = df1.sort_values(by=['q'], ascending=True)
    xgrid = xgrid.sort_values(ascending=True)

    # Extract q and I columns from df1
    q_values = df1['q'].to_numpy()
    i_values = df1['I'].to_numpy()

    # Interpolate I values based on xgrid['q']
    interpolated_i = np.interp(xgrid.to_numpy(), q_values, i_values)

    # Combine xgrid with interpolated values
    result = pd.concat([xgrid, pd.DataFrame({'I': interpolated_i})], axis=1)

    return result
