'''
author: @slothfulwave612

In this module we will create random data.
'''

## import required modules/packages
from random import randint

def data():
    '''
    Function to create random data.

    Returns:
    train_samples -- list, of samples containing ages.
    train_labels -- list, of labels containing 0 or 1.
                    0 -- individual did not experienced any effects from the drug
                    1 -- individual did experienced effects from the drug
    '''

    train_samples = []          ## input data
    train_labels = []           ## target data

    ## outliers
    for _ in range(50):
        ## the ~5% young individuals who did experienced side effects
        random_young = randint(13, 64)
        train_samples.append(random_young)      ## the age of the individual [13, 64]
        train_labels.append(1)                  ## 1 -- experienced side effects

        ## the ~5% old individuals who did not experienced side effects
        random_old = randint(65, 100)
        train_samples.append(random_old)        ## the age of the individual [65, 100]
        train_labels.append(0)                  ## 0 -- did not experienced side effects

    for _ in range(1000):
        ## the 95% young individuals who did not experienced side effects
        random_young = randint(13, 64)
        train_samples.append(random_young)      ## the age of the individual [13, 64]
        train_labels.append(0)                  ## 0 -- did not experienced side effects

        ## the 95% old individuals who did experienced side effects
        random_old = randint(65, 100)           
        train_samples.append(random_old)        ## the age of individual [65, 100]
        train_labels.append(1)                  ## 1 -- did experienced side effects
    
    return train_samples, train_labels
