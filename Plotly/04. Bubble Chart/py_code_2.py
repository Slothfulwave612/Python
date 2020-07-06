'''
Author: @slothfulwave612

Plotting a bubble chart.

Data: mpg
'''

## import necessary libraries
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go

## reading in the data
df = pd.read_csv('../Data/mpg.csv')

## using it will plot the data
df['final_name'] = df['name'] + '-' + df['model_year'].astype(str)


## create data list
data = [go.Scatter(
    x = df['horsepower'],
    y = df['mpg'],
    text = df['final_name'],
    mode = 'markers',
    marker = dict(size=1.5 * df['cylinders'])
)]

## create layout
layout = go.Layout(
    title = 'Vehicle mpg vs horsepower',
    xaxis = dict(title = 'horsepower'),
    yaxis = dict(title = 'mpg'),
    hovermode = 'closest'
)

## create fig object
fig = go.Figure(data=data, layout=layout)

## plot and save
pyo.plot(fig, filename="temp-plot.html")
