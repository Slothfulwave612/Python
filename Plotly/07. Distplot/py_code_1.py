'''
Author: @slothfulwave612

A simple distplot.
'''

## necessary packages
import pandas as pd
import plotly.offline as pyo
import plotly.figure_factory as ff

## generating random data
x = np.random.randn(1000)
hist_data = [x]
hist_label = ['distplot']

## create fig object
fig = ff.create_distplot(hist_data, hist_label)

## plot and save
pyo.plot(fig, filename="temp-plot.html")
