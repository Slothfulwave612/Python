'''
Author: @slothfulwave612

A simple histogram
'''

## necessary packages
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go

## read in the dataset
df = pd.read_csv('../Data/mpg.csv')

## create data list
data = [go.Histogram(
    x = df['mpg']
)]

## create layout object
layout = go.Layout(
    title = 'Miles per Gallon Frequencies of<br>\
    1970\'s Era Vehicles'
)

## create a figure object
fig = go.Figure(data=data, layout=layout)

## plot and save
pyo.plot(fig, filename="temp-plot.html")
