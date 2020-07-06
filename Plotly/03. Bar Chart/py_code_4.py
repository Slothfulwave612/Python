'''
Author: @slothfulwave612

A horizontal bar chart.

Data: mocksurvey
'''

## import necessary libraries
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go

## reading in the data
df = pd.read_csv('../Data/mocksurvey.csv', index_col=0)

## create a data list
data = [go.Bar(
    y = df.index,
    x = df[response],
    orientation='h',
    name = response
) for response in df.columns]

## create layout object
layout = go.Layout(
    title = 'Mock Survey',
    barmode = 'stack'
)

## create fig object
fig = go.Figure(data=data, layout=layout)

## plotting and saving
pyo.plot(fig, filename="temp-plot.html")
