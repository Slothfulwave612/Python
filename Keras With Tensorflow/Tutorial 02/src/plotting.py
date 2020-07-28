'''
author: @slothfulwave612

Python module for plotting.
'''

## necessary library
from matplotlib import pyplot as plt

def plot_images(imgs, labels):
    '''
    Function for plotting images.

    Argument:
    imgs -- array of images containing pixel values.
    label -- image label
    '''

    ## create subplots
    _, axes = plt.subplots(nrows=1, ncols=10, figsize=(20,20))

    ## plot images
    for img, label, ax  in zip(imgs, labels, axes):
        ax.imshow(img, cmap='Greys')

        if list(label) == [0.0, 1.0]:
            ax.set_title('Dog')
        else:
            ax.set_title('Cat')
        
        ax.axis('off')

    plt.tight_layout()
