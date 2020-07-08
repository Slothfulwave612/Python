'''
Author: @slothfulwave612

Box plot to compares sample distributions of three-letter words in
the works of Quintus Curtius Snodgrass and Mark Twain
'''

## importing libraries
import plotly.offline as pyo
import plotly.graph_objs as go

## the data points
snodgrass = [.209,.205,.196,.210,.202,.207,.224,.223,.220,.201]
twain = [.225,.262,.217,.240,.230,.229,.235,.217]

## creating traces for both the arrays
trace_snodgrass = go.Box(
    y = snodgrass,
    name = 'Snodgrass'
)

trace_twain = go.Box(
    y = twain,
    name = 'Twain'
)

## creating a data list
data = [trace_snodgrass, trace_twain]

## creating layout object
layout = go.Layout(
    title = 'Comparison of three-letter word frequencies<br>\
between Quintus Curtius Snodgrass and Mark Twain'
)

## creating fig object
fig = go.Figure(data=data, layout=layout)

## plotting and saving
pyo.plot(fig, filename="temp-plot.html")
