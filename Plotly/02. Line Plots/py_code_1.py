'''
author: @slothfulwave612

Creating line plots.

Here we will show three plots in the same file.
Plot 1 :- Top, scatter plot.
Plot 2 :- In between, line plot.
Plot 3 :- Bottom, line plot with markers.
'''

## importing necessary packages
import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go

## creating random data
np.random.seed(42)
x_values = np.linspace(0, 1, 100)       ## 100 evenly spaced values
y_values = np.random.randn(100)         ## 100 random values <-- normal dist

## creating traces
trace_0 = go.Scatter(
    x = x_values,
    y = y_values + 5,
    mode = 'markers',
    name = 'markers'
)   ## for plotting scatter plot

trace_1 = go.Scatter(
    x = x_values,
    y = y_values,
    mode = 'lines',
    name = 'lines'
)   ## for plotting line plot

trace_2 = go.Scatter(
    x = x_values,
    y = y_values - 5,
    mode = 'lines+markers',
    name = 'lines+markers'
)   ## for plottinh line plot with markers

## creating data list
data = [trace_0, trace_1, trace_2]

## creating layout object
layout = go.Layout(
    title = 'Chart showing three different modes',
    hovermode='closest'
)

## creating figure object
fig = go.Figure(data=data, layout=layout)

## plotting and saving the figure
pyo.plot(fig, filename="line_1.html")
