'''
Author: @slothfulwave612

A distplot having multiple data points.
'''

## necessary packages
import pandas as pd
import plotly.offline as pyo
import plotly.figure_factory as ff

## generating random data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200) 
x3 = np.random.randn(200) + 2
x4 = np.random.randn(200) + 4

hist_data = [x1, x2, x3, x4]
group_labels = 'Group_1,Group_2,Group_3,Group_4'.split(',')

## create figure object
fig = ff.create_distplot(hist_data, group_labels)

## plot and save
pyo.plot(fig, filename="temp-plot.html")
