'''
author: @slothfulwave612

Python module for running scripts.
'''

## import necessary modules
import create_data as cd
import processing as pc
import train as tn
import test as tt

import tensorflow as tf

## loading the data
train_samples, train_labels = cd.data()

## testing and setting gpu  --> run only if you are using a gpu
## to see the physical devices
physical_devices = tf.config.experimental.list_physical_devices('GPU')

## will display number of gpu integrated with tensorflow
print(f'Number of GPU availabe: {len(physical_devices)}')

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

'''
## Running basic model
scaled_train_samples, train_labels = pc.process(train_samples, train_labels)
model = tn.create_model()
model = tn.train_model(model, x_train=scaled_train_samples, y_train=train_labels)
'''

'''
## Running model with train set and valid set(manual-defined)
scaled_train_samples, train_labels, scaled_valid_samples, valid_labels = pc.train_valid_split(x=train_samples, y=train_labels, per=0.1)
model = tn.create_model()
model = tn.train_valid_model(
    model=model, 
    x_train=scaled_train_samples, 
    y_train=train_labels,
    valid_set=(scaled_valid_samples, valid_labels))
'''

'''
## Running model with train set and valid set(using Keras)
scaled_train_samples, train_labels = pc.process(train_samples, train_labels)
model = tn.create_model()
model = tn.train_valid_model(
    model=model, 
    x_train=scaled_train_samples, 
    y_train=train_labels,
    per=0.1)
'''

'''
## Training mode on train-data and testing it on test data, no validation data is used here
## (you can create one just by passing a parameter, as above)

## training-data
scaled_train_samples, train_labels = pc.process(train_samples, train_labels)            

## test-data
test_samples, test_labels = cd.test_data()
scaled_test_samples, test_labels = pc.process(test_samples, test_labels)

model = tn.create_model()
model = tn.train_model(model, x_train=scaled_train_samples, y_train=train_labels)

predictions = tt.predict(model, scaled_test_samples, test_labels)
predictions = pc.process_predicitons(predictions)
classes = ['no-side-effects', 'had-side-effects']
tt.plot_confusion_matrix(classes, test_labels, predictions, acc=True, savefig='confusion_matrix_1')
'''

## Training mode on train-data, validation on validation-set and testing it on test data, 
## training-data
scaled_train_samples, train_labels = pc.process(train_samples, train_labels)            

## test-data
test_samples, test_labels = cd.test_data()
scaled_test_samples, test_labels = pc.process(test_samples, test_labels)

model = tn.create_model()
model = tn.train_valid_model(model, scaled_train_samples, train_labels, per=0.1)

predictions = tt.predict(model, scaled_test_samples, test_labels)
predictions = pc.process_predicitons(predictions)
classes = ['no-side-effects', 'had-side-effects']
tt.plot_confusion_matrix(classes, test_labels, predictions, acc=True, savefig='confusion_matrix_2')