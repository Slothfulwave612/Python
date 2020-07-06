'''
Author: @slothfulwave612

Basic bar chart in plotly. We'll plot 2018 Winter Olympic Medals won by country.
'''

## import necessary libraries
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go

## reading in the data
df = pd.read_csv('Data/2018WinterOlympics.csv')

df.head()

## creating a data list
data = [go.Bar(
    x = df['NOC'],
    y = df['Total'],
)]

## creating layout object
layout = go.Layout(title = '2018 Winter Olympic Medals by Country')

## creating figure object
fig = go.Figure(data=data, layout=layout)

## plotting and saving
pyo.plot(fig, filename='test.html')
