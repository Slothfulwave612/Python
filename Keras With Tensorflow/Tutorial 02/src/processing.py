'''
author: @slothfulwave612

Python module for preprocessing the data.
'''

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def image_processing(train_path, valid_path, test_path, class_names, batch_size, image_size, shuffle=True):
    '''
    Function for processing our images based on the values passed.

    Arguments:
    train_path -- str, path to the train directory.
    valid_path -- str, path to the valid directory.
    test_path -- str, path to the test directory.
    class_names -- list, of class name.
    batch_size -- int, total number of images in one batch.
    image_size -- tuple, containing int pixel values.
    shuffle -- bool, for shuffling the dataset.

    Returns:
    train_batches, valid_batches, test_batches -- preprocessed images for each set.
    '''

    ## creating train-batches
    train_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory = train_path,
        classes = class_names,
        batch_size = batch_size,
        target_size = image_size,
        shuffle = shuffle
    )

    ## creating valid-batches
    valid_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory = valid_path,
        classes = class_names,
        batch_size = batch_size,
        target_size = image_size,
        shuffle = shuffle
    )

    ## creating test-batches
    test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory = test_path,
        classes = class_names,
        batch_size = batch_size,
        target_size = image_size,
        shuffle = shuffle
    )

    return train_batches, valid_batches, test_batches

        