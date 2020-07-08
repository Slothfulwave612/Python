'''
Author: @slothfulwave612

Plotting a simple box plot.(includes jitter and pointpos)
'''

## importing libraries
import plotly.offline as pyo
import plotly.graph_objs as go

# set up an array of 20 data points, with 20 as the median value
y = [1,14,14,15,16,18,18,19,19,20,20,23,24,26,27,27,28,29,33,54]

## creating a data list
data = [go.Box(
    y = y,
    boxpoints = 'all',    ## to show all points
    jitter = 0.4,         ## spread the data points
    pointpos = 1.5        ## offset them to the right of the box
)]

## plotting and saving
pyo.plot(data, filename="temp-plot.html")
