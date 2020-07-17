'''
author: @slothfulwave612

In this module we will perform the pre-processing of the data.
'''

## import required packages/modules
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

def process(train_samples, train_labels):
    '''
    Function to perform pre-processing.

    Arguments:
    train_samples -- list, containing input data.
    train_labels -- list, containing target data.

    Returns:
    scaled_train_samples -- numpy array, containing normalized values.
    train_labels -- numpy array, containing target data.
    '''
    ## convert list to numpy array
    train_samples = np.array(train_samples)
    train_labels = np.array(train_labels)

    ## shuffle the data
    train_samples, train_labels = shuffle(train_samples, train_labels)

    ## MinMaxScalar object
    scaler = MinMaxScaler(feature_range=(0, 1))

    ## scale train_samples
    scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))

    return scaled_train_samples, train_labels

