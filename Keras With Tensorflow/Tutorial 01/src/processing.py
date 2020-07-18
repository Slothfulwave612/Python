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

def train_valid_split(x, y, per):
    '''
    Function for creating a validation set.

    Arguments:
    x -- list, containing input data.
    y -- list, containing target data.
    per -- float, fraction value for splitting.
           e.g. if per=0.1, that means 10% of data will be in validation set rest in train set.

    Returns:
    train_samples -- numpy array, containing normalized values for input data(for training).
    train_labels -- numpy array, containing target data(for training).
    valid_samples -- numpy array, containing normalized values for input data(for validation).
    valid_labels -- numpy array, containing target data(for validation).
    '''
    ## convert list to numpy arrays
    x = np.array(x)
    y = np.array(y)

    ## shuffle both arrays --- shuffling reduce bias
    x, y = shuffle(x, y)

    ## getting length for valid arrays
    val_len = int(len(x) * per)

    ## making validation set
    valid_samples = x[:val_len]
    valid_labels = y[:val_len]

    ## making training set
    train_samples = x[val_len:]
    train_labels = y[val_len:]

    ## MinMaxScalar object
    scaler = MinMaxScaler(feature_range=(0, 1))

    ## scale train_samples
    scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))

    ## scale valid_samples
    scaled_valid_samples = scaler.fit_transform(valid_samples.reshape(-1, 1))

    return scaled_train_samples, train_labels, scaled_valid_samples, valid_labels