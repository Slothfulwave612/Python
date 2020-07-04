'''
author: @slothfulwave612

Plotting a simple scatter plot using Plotly.

Step 1: Create random data.
Step 2: Create a data list(containing Scatter object defined in graph_objs).
Step 3: Plot and save the file in .html format.
'''

## importing necessary libraries
import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go

## creating random data points
np.random.seed(42)
random_x = np.random.randint(1, 101, 100)
random_y = np.random.randint(1, 101, 100)

## creating data list
data = [go.Scatter(
    x = random_x,
    y = random_y,
    mode = 'markers'
)]

## saving the plot in a html file
pyo.plot(data, filename='scatter_1.html')
