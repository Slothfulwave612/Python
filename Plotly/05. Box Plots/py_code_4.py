'''
Author: @slothfulwave612

Taking two random samples of different sizes from 'rings' field and
plot the box plot.

Data: abalone
'''

## importing libraries
import numpy as np
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go

## reading the dataset
df = pd.read_csv('../Data/abalone.csv')

dist_1 = np.random.choice(df['rings'], 50, replace=False)
dist_2 = np.random.choice(df['rings'], 15, replace=False)

## making traces for dist_1 and dist_2
trace_dist_1 = go.Box(
    y = dist_1,
    name = 'dist_1'
)

trace_dist_2 = go.Box(
    y = dist_2,
    name = 'dist_2'
)

## creating data list
data = [trace_dist_1, trace_dist_2]

## adding a layout
layout = go.Layout(
    title = 'Plotting 2 box plots'
)

## creating fig object
fig = go.Figure(data=data, layout=layout)

## plotting and saving
pyo.plot(fig, filename="temp-plot.html")
