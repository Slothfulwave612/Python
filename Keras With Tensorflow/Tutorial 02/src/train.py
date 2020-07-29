'''
author: @slothfulwave612

Python module for training.
'''
## necessary packages/modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

def create_model():
    '''
    Function fo creating a simple sequential convolutional model.

    Returns:
    model -- Sequential model
    '''

    ## create model
    model = Sequential(
      [
          Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
          MaxPool2D(pool_size=(2, 2), strides=2),
          Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
          MaxPool2D(pool_size=(2, 2), strides=2),
          Flatten(),
          Dense(units=2, activation='softmax')
      ]
    )

    ## compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train(model, train_batches, valid_batches):
    '''
    Function for training our model, and evaluating on validation set.

    Arguments:
    model -- Sequential model.
    train_batches -- our training batches.
    valid_batches -- our validation batches.

    Returns:
    model -- Sequential model
    '''

    ## training the model
    model.fit(
        x = train_batches, 
        steps_per_epoch = len(train_batches),
        validation_data = valid_batches,
        validation_steps = len(valid_batches),
        epochs = 10,
        verbose = 2
    )

    return model