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

def test_data():
    '''
    Function for creating test-set data.

    Returns:
    test_samples -- list, of samples containing ages.
    test_labels -- list, of labels containing 0 or 1.
                   0 -- individual did not experienced any effects from the drug
                   1 -- individual did experienced effects from the drug
    '''

    test_samples = []            ## input data
    test_labels = []             ## target values

    ## outliers
    for _ in range(10):
        ## the ~5% young individuals who did experienced side effects
        random_young = randint(13, 64)
        test_samples.append(random_young)      ## the age of the individual [13, 64]
        test_labels.append(1)                  ## 1 -- experienced side effects

        ## the ~5% old individuals who did not experienced side effects
        random_old = randint(65, 100)
        test_samples.append(random_old)        ## the age of the individual [65, 100]
        test_labels.append(0)                  ## 0 -- did not experienced side effects

    for _ in range(200):
        ## the 95% young individuals who did not experienced side effects
        random_young = randint(13, 64)
        test_samples.append(random_young)      ## the age of the individual [13, 64]
        test_labels.append(0)                  ## 0 -- did not experienced side effects

        ## the 95% old individuals who did experienced side effects
        random_old = randint(65, 100)           
        test_samples.append(random_old)        ## the age of individual [65, 100]
        test_labels.append(1)                  ## 1 -- did experienced side effects
    
    return test_samples, test_labels