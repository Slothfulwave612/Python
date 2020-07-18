'''
author: @slothfulwave612

Python module for running scripts.
'''

## import necessary modules
import create_data as cd
import processing as pc
import train as tn

## loading the data
train_samples, train_labels = cd.data()

## testing and setting gpu  --> run only if you are using a gpu
tn.test_gpu()

'''
## Running basic model
scaled_train_samples, train_labels = pc.process(train_samples, train_labels)
model = tn.create_model()
model = tn.train_model(model, x_train=scaled_train_samples, y_train=train_labels)
'''

## Running model with train set and valid set(manual-defined)
'''
scaled_train_samples, train_labels, scaled_valid_samples, valid_labels = pc.train_valid_split(x=train_samples, y=train_labels, per=0.1)
model = tn.create_model()
model = tn.train_valid_model(
    model=model, 
    x_train=scaled_train_samples, 
    y_train=train_labels,
    valid_set=(scaled_valid_samples, valid_labels))
'''

# ## Running model with train set and valid set(using Keras)
'''    
scaled_train_samples, train_labels = pc.process(train_samples, train_labels)
model = tn.create_model()
model = tn.train_valid_model(
    model=model, 
    x_train=scaled_train_samples, 
    y_train=train_labels,
    per=0.1)
'''