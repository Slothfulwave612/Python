'''
author: @slothfulwave612

In this module we will be implementing our Sequential model.
'''

## import required packages/libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense

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
