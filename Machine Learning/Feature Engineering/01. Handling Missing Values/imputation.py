'''
author: @slothfulwave612

Python module for different imputation technique.
'''

## import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def per_null(df, col):
    '''
    Function to find the percentage of 
    null values for a column in a dataframe.

    Arguments:
    df -- dataframe object.
    col -- str, name of the column.

    Returns:
    percent -- float, percentage of null values
    '''
    ## for null assign 0 else assign 1
    null_values = np.where(df[col].isnull(), 1, 0)

    percent = null_values.mean()

    return percent

def central_tendency_fill(df, col, centralT='mean'):
    '''
    Function to make a new column in the dataframe and
    filling with the mean/meadian/mode of the variable.

    Arguments:
    df -- dataframe object.
    col -- str, column name.
    centralT -- central tendency, default is 'mean'.
                Values can be 'mean', 'median' or 'mode'

    Returns:
    df -- dataframe object.
    '''
    if centralT == 'mean':
        ## calculate mean
        val = df[col].mean()
    
    elif centralT == 'median':
        ## calculate median
        val = df[col].median()
    
    elif centralT == 'mode':
        ## calculate mode
        val = df[col].mode().values[0]

    ## create a new column 
    df[col + f'_{centralT}'] = df[col].fillna(val)

    return df

def plot_values(df, cols):
    '''
    Function to plot comparison between two column values.

    Arguments:
    df -- dataframe object.
    cols -- list, of column names
    '''
    ## create subplot
    _, ax = plt.subplots(figsize=(8,6))

    for col in cols:
        df[col].plot(kind='kde', ax=ax)

    ## labels
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='best')

def random_imputation(df, col) :
    '''
    Function to fill NaN values using random-sample-imputation technique.

    Arguments:
    df -- dataframe object.
    col -- column name.

    Returns:
    df -- dataframe object
    '''
    ## create a column
    df[col + '_random'] = df[col]

    ## pick random values from the dataframe
    ## number of random values to be picked == total number of NaN values in that column
    random_values = df[col].dropna().sample(
        n = df[col].isnull().sum(),
        random_state = 42
    )

    ## random-values should have same index as null values
    random_values.index = df.loc[df[col].isnull()].index

    ## fill NaN values in new column we made
    df.loc[df[col].isnull(), col + '_random'] = random_values

    return df

def capture_nan(df, col):
    '''
    Function to capture nan values.

    Arguments:
    df -- dataframe object.
    col -- the column name.

    Returns:
    df -- dataframe object.
    '''
    df[col + '_NaN'] = np.where(df[col].isnull(), 1, 0)

    return df

def end_of_dist(df, col):
    '''
    Function to fill NaN values with end of distribution value.

    Arguments:
    df -- dataframe object.
    col -- the column name.

    Returns:
    df -- dataframe object.
    '''
    extreme_val = df[col].mean() + (3 * df['Age'].std())

    df[col + '_end_of_dist'] = df[col].fillna(extreme_val)

    return df
