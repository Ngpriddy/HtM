"""
Collection of function useful for HackTheMachine 2019

Author: Zachary Davis
Date: 09-03-2019
"""

import pandas as pd

def date_time_convert(df, path_to_csv="converted_data.csv"):
    """
    Function to convert the inconsistent date/time format between the MSP and
    MAF data into a standard format for sorting purposes. Also saves converted
    data to .csv file with specified name

    Parameters
    ----------
    df (Pandas dataframe)
        The raw dataframe
    path_to_csv (string), default 'converted_data.csv' 
        Path to save the converted dataframe

    Returns
    -------
    converted_df (Pandas dataframe)
        Dataframe with converted date/time format
    """

    converted_df = df

    # Determine if MSP or MAF data is being converted
    if df.columns[0] == 'Job Code':
        # Convert appropriate columns of MSP data
        converted_df['Received Date'] = pd.to_datetime(df['Received Date'], format="%m/%d/%y").astype(str)
        converted_df['Completion Date'] = pd.to_datetime(df['Completion Date'], format="%m/%d/%y").astype(str)
    else:
        # Convert appropriate columns of MAF data
        converted_df['ZULU_TIME'] = pd.to_datetime(df['ZULU_TIME'].str[1:], format="%d-%b-%Y %H:%M:%S:%f").astype(str)

    # Save converted dataframe to filename specified by function call
    converted_df.to_csv(path_to_csv)

    return converted_df
