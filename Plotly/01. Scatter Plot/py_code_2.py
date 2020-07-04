'''
author: @slothfulwave612

Plotting a simple scatter plot using Plotly and adding labels.

Step 1: Create random data.
Step 2: Create a data list(containing Scatter object defined in graph_objs).
Step 3: Create a layout for adding title and labels
Step 4: Create a figure object to combine both data and layout.
Step 5: Plot and save the file in .html format.
'''

## importing necessary packages
import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go

## creating random data.
np.random.seed(42)
random_x = np.random.randint(1, 101, 100)
random_y = np.random.randint(1, 101, 100)

## creating data list
data = [go.Scatter(
    x = random_x,
    y = random_y,
    mode = 'markers'
)]

## adding a layout
layout = go.Layout(
    title='A Scatterplot Sample',               ## graph title
    xaxis=dict(title='Some random x-values'),   ## x label
    yaxis=dict(title='Some random y-values'),   ## y label
    hovermode='closest'  ## handles multiple points landing on the same vertical
)

## creating a figure object
fig = go.Figure(data=data, layout=layout)

## saving the plot
pyo.plot(fig, filename='scatter_2.html')
