'''
author: @slothfulwave612

In this module we will be implementing our Sequential model.
'''

## import required packages/libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam

def test_gpu():
    '''
    Function to test whether gpu is working or not with tensorflow.
    '''
    ## to see the physical devices
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    ## will display number of gpu integrated with tensorflow
    print(f'Number of GPU availabe: {len(physical_devices)}')

    ## to enable memory growth for a PhysicalDevice
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def create_model():
    '''
    Function to create Sequential model.

    Returns:
    model -- Sequential model.
    '''

    ## Sequential Model
    model = Sequential(layers=[
        Dense(units=16, input_shape=(1,), activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])

    return model

def train_model(model, x_train, y_train):
    '''
    Function to compile and train the model.

    Argument:
    model -- Sequential model.
    x_train -- input data.
    y_train -- target labels.

    Returns:
    model -- Sequential model.
    '''
    ## compiling the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    ## training the model
    model.fit(x=x_train, y=y_train, batch_size=10, epochs=30, verbose=2)

    return model

def train_valid_model(model, x_train, y_train, valid_set=None, per=None):
    '''
    Function to compile and train the model(with validation set).

    Arguments:
    model -- Sequential model.
    x_train -- input data.
    y_train -- target labels.
    valid_set -- validation set, default set to None --> tuple (valid_samples, valid_targets).
    per -- float, fraction value for splitting.
           e.g. if per=0.1, that means 10% of data will be in validation set rest in train set.

    Returns:
    model -- Sequential model.
    '''
    ## compiling the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    ## training the model and evaluating on validation set
    if valid_set:
        model.fit(x=x_train, y=y_train, validation_data=valid_set, batch_size=10, epochs=30, verbose=2)
    else:
        model.fit(x=x_train, y=y_train, validation_split=per, batch_size=10, epochs=30, verbose=2)

    return model