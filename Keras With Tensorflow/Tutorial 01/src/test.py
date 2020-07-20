'''
author: @slothfulwave612

Python module for testing, evaluation and saving the model.
'''

## importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model, model_from_json
import itertools
import json

def predict(model, test_samples, test_labels):
    '''
    Function for prediction.

    Arguments:
    model -- a Sequential model.
    test_samples -- numpy array, input data(scaled).
    test_labels -- numpy array, target values.

    Returns:
    predictions -- numpy array, of predicted values.
    '''

    ## use predict function
    predictions = model.predict(x=test_samples, batch_size=10, verbose=0)

    return predictions

def plot_confusion_matrix(classes, y_true, y_pred, cmap=plt.cm.Blues, acc=False, savefig=None):
    '''
    Function to plot a confusion matrix based on 
    y_true value and y_pred values.

    Arguments:
    classes -- list, of classes.
    y_true -- numpy array, of true values.
    y_pred -- numpy array, of predicted values.
    cmap -- color map, default --> cm.Blues
    acc -- True if you want to see the accuracy
    '''
    ## making confusion matrix
    cnf_mat = confusion_matrix(y_true, y_pred)

    ## plotting our confusion matrix
    plt.imshow(cnf_mat, interpolation='nearest', cmap=cmap)

    ## labeling our confusion matrix
    plt.title('Confusion Matrix')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    ## creating the colorbar
    plt.colorbar()

    ## prinitng the accuracy
    if acc == True:
        print(f'Acuuracy Score: {accuracy_score(y_true, y_pred)}')

    thresh = cnf_mat.max() / 2.
    for i, j in itertools.product(range(cnf_mat.shape[0]), range(cnf_mat.shape[1])):
        plt.text(j, i, cnf_mat[i, j],
            horizontalalignment="center",
            color="white" if cnf_mat[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if savefig:
        plt.savefig(f'../plots/{savefig}.png', dpi=300)
        plt.close()
    
    else:
        plt.show()

def save_model(model, model_name, value):
    '''
    Function to save the model.

    Arguments:
    model -- a Sequential model.
    model_name -- str, the name of the model.
    value -- int value.
             0 -- saving the model entirely.
             1 -- saving only the architecture.
             2 -- saving only the weights.
    '''

    if value == 0:
        ## saving the entire model
        model.save(f'../models/{model_name}')

    elif value == 1:
        ## saving only the architecture
        json_val = model.to_json()

        with open(f'../models/{model_name}', 'w') as model_file:            
            model_file.write(json_val)

    elif value == 2:
        ## saving only the weights
        model.save_weights(f'../models/{model_name}')
    
def load_models(model_name, value, model=None):
    '''
    Function to load model.

    Arguments:
    model_name -- str, the name of the model.
    value -- int value.
             0 -- loading the model entirely.
             1 -- loading only the architecture.
             2 -- loading only the weights.
    model -- a model should be passed only when value == 2, i.e. when loading weights

    Returns:
    model -- the save model.
    '''
    if value == 0:
        ## loading the entire model
        model = load_model(f'../models/{model_name}')
    
    elif value == 1:
        ## loading only the architecture
        with open(f'../models/{model_name}', 'r') as model_file:         
            model = model_file.read()
        model = model_from_json(model)

    elif value == 2:
        ## loading only the weights
        model.load_weights(f'../models/{model_name}')
    
    return model