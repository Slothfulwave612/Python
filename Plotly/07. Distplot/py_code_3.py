'''
Author: @slothfulwave612

A distplot that compares the petal lengths of each class.

Fields: 'sepal_length','sepal_width','petal_length','petal_width','class'
Classes: 'Iris-setosa','Iris-versicolor','Iris-virginica'
'''

## necessary packages
import pandas as pd
import plotly.offline as pyo
import plotly.figure_factory as ff

## read in the dataset
df = pd.read_csv('../Data/iris.csv')

## creating traces for each class
trace_setosa = df.loc[df['class'] == 'Iris-setosa', 'petal_length']
trace_versicolor = df.loc[df['class'] == 'Iris-versicolor', 'petal_length']
trace_virginica = df.loc[df['class'] == 'Iris-virginica', 'petal_length']

hist_data = [trace_setosa, trace_versicolor, trace_virginica]
group_labels = ['Setosa', 'Versicolor', 'Virginica']

## create fig object
fig = ff.create_distplot(hist_data, group_labels)

## plot and save
pyo.plot(fig, filename="temp-plot.html")
