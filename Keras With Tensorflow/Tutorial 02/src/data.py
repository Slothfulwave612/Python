'''
author: @slothfulwave612

Python module for managing dataset.
'''

## necessary packages/modules
import os
import glob
import shutil
import random

def create_split(path='input'):
    '''
    Function for creating train-valid-test split.

    Argument:
    path -- str, containing the path of the directory where the images are stored.
    '''
    ## changing the path
    os.chdir(path)

    ## creating split
    if os.path.isdir('train') is False:

        ## making required directories
        os.mkdir('train')
        os.mkdir('valid')
        os.mkdir('test')
        os.mkdir('train/dog')
        os.mkdir('train/cat')
        os.mkdir('valid/dog')
        os.mkdir('valid/cat')
        os.mkdir('test/dog')
        os.mkdir('test/cat')

        ## moving 1000 random cat images to train/cat
        print('Creating Training Data...', end='')
        for i in random.sample(glob.glob('cat*'), 1000):
            shutil.move(i, 'train/cat')

        ## moving 1000 random dog images to train/dog
        for i in random.sample(glob.glob('dog*'), 1000):
            shutil.move(i, 'train/dog')
        print('Completed!!!')

        ## moving 300 random cat images to valid/cat
        print('Creating Validation Data...', end='')
        for i in random.sample(glob.glob('cat*'), 300):
            shutil.move(i, 'valid/cat')
        
        ## moving 300 random dog images to valid/dog
        for i in random.sample(glob.glob('dog*'), 300):
            shutil.move(i, 'valid/dog')
        print('Completed!!!')
        
        ## moving 100 random cat images to test/cat
        print('Creating Test Data...', end='')
        for i in random.sample(glob.glob('cat*'), 100):
            shutil.move(i, 'test/cat')
        
        ## moving 100 random cat images to test/dog
        for i in random.sample(glob.glob('cat*'), 100):
            shutil.move(i, 'test/dog')
        print('Completed!!!')
        
    os.chdir('..')

if __name__ == '__main__':
    ## create data splits
    create_split()
