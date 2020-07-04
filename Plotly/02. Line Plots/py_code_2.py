'''
author: @slothfulwave612

Creating line plot from a DataFrame.
'''

## importing necessary packages
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go

## reading in the data
df = pd.read_csv('Data/population.csv', index_col=0)

## creating data list
data = [go.Scatter(
    x = df.columns,
    y = df.loc[name],
    mode = 'lines+markers',
    name = name
) for name in df.index]

## creating layout object
layout = go.Layout(
    title = 'Population Estimates of six england states'
)

## creating figure object
fig = go.Figure(data=data, layout=layout)

## plotting and saving
pyo.plot(fig, filename="line_2.html")
