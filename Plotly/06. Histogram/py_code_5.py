'''
Author: @slothfulwave612

A histogram that plots the 'length' field from the Abalone dataset.
'''

## necessary packages
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go

## read in the dataset
df = pd.read_csv('../Data/abalone.csv')

## create data list
data = [go.Histogram(
    x = df['length'],
    xbins = dict(start=0, end=1, size=0.02)
)]

## add a layout
layout = go.Layout(
    title = 'Length of the field'
)

## create a fig object
fig = go.Figure(data=data, layout=layout)

## plot and save
pyo.plot(fig, filename="temp-plot.html")
