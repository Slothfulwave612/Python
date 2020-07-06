'''
Author: @slothfulwave612

A grouped bar chart. Groups(Gold, Silver and Bronze)

Data: 2018WinterOlympics
'''

## import necessary libraries
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go

## reading in the data
df = pd.read_csv('../Data/2018WinterOlympics.csv')

## creating traces for goal, silver and bronze
trace_gold = go.Bar(
    x = df['NOC'],
    y = df['Gold'],
    name = 'Gold',
    marker = dict(color='#FFD700')
)

trace_silver = go.Bar(
    x = df['NOC'],
    y = df['Silver'],
    name = 'Silver',
    marker = dict(color='#9EA0A1')
)

trace_bronze = go.Bar(
    x = df['NOC'],
    y = df['Bronze'],
    name = 'Bronze',
    marker = dict(color='#CD7F32')
)

## creating data list
data = [trace_gold, trace_silver, trace_bronze]

## creating layout object
layout = go.Layout(title='2018 Winter Olympic Medals by Country')

## creating figure object
fig = go.Figure(data=data, layout=layout)

## plotting and saving
pyo.plot(fig, filename='test.html')
