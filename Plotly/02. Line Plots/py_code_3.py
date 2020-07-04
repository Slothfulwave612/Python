'''
author: @slothfulwave612

Using file 2010YumaAZ.csv ploting a line chart showing seven days worth of
temperature data on one graph
'''

## importing necessary packages
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go

days = ['TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY','MONDAY']

## reading in the data
df = pd.read_csv('Data/2010YumaAZ.csv')

data = []
for day in days:
    ## creating traces
    traces = go.Scatter(
        x = df.loc[df['DAY'] == day, 'LST_TIME'],
        y = df.loc[df['DAY'] == day, 'T_HR_AVG'],
        mode = 'lines+markers',
        name = day
    )
    ## appending traces in data list
    data.append(traces)

## creating layout object
layout = go.Layout(
    title='Temperature',
    hovermode='closest'
)

## creating figure object
fig = go.Figure(data=data, layout=layout)

## plotting and saving
pyo.plot(fig, filename='line_3.html')
