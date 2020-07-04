'''
author: @slothfulwave612

Creating scatter plot using normal and uniform distribution.

Step 1: Create random data.
Step 2: Create a data list(containing Scatter object defined in graph_objs).
Step 3: Create a layout for adding title and labels.
Step 4: Create a figure object to combine both data and layout.
Step 5: Plot and save the file in .html format.
'''

## importing necessary packages
import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go

## creating random data.
np.random.seed(42)
random_x = np.random.randn(1000)        ## normal distribution
random_y = np.random.rand(1000)         ## uniform distribution

## creating data list
data = [go.Scatter(
    x = random_x,
    y = random_y,
    mode = 'markers',
    marker = dict(
        size = 15,
        color = 'rgb(123,212,32)',
        symbol = 'square'
    )
)]

## creating layout object
layout = go.Layout(
    title = 'Scatterplot',
    xaxis = dict(title = 'Values from normal distribution'),
    yaxis = dict(title = 'Values from uniform distribution'),
    hovermode='closest'
)

## creating a figure object
fig = go.Figure(data=data, layout=layout)

## saving the plot
pyo.plot(fig, filename="scatter_4.html")
