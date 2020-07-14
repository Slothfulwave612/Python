'''
Author: @slothfulwave612

A histogram for comparing heights by gender.
'''

## necessary packages
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go

## read in the dataset
df = pd.read_csv('../Data/arrhythmia.csv')

## creating traces for height of male and female
trace_male = go.Histogram(
    x = df.loc[df['Sex'] == 0, 'Height'],
    opacity = 0.75,
    name = 'Male'
)

trace_female = go.Histogram(
    x = df.loc[df['Sex'] == 1, 'Height'],
    opacity = 0.75,
    name = 'Female'
)

## create data list
data = [trace_male, trace_female]

## create layout object
layout = go.Layout(
    title = 'Height Comparison by Gender',
    barmode = 'overlay'
)

## create figure object
fig = go.Figure(data=data, layout=layout)

## plot and save
pyo.plot(fig, filename="temp-plot.html")
