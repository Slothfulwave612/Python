## Convolutional Neural Network

## Contents

* [Overview](#overview)

* [Directory Tree](#directory-tree)

* [Organize The Data](#organize-the-data)

* [Processing The Data](#processing-the-data)

## Directory Tree

```
|
```

## Overview

* Here in this tutorial we will take a look on how to build a CNN using Keras integrated with TensorFlow.

* We will be using a data set from the [Kaggle Dogs Versus Cats competition](https://www.kaggle.com/c/dogs-vs-cats/data). 

* Prerequisite: Understanding of Convolutional Neural Networks.

## Organize The Data

* We now need to organize the directory structure on disk to hold the data set. 

* We'll manually do some parts of the organization, and programmatically do the rest.

* First of all extract all the images from `train.zip` into `input` directory, i.e. `input` directory should contain all the images.

* After that run the below script to make three directories `train`, `valid` and `test` each having two subfolders `cat` and `dog` where images will be moved.

  ```python
  ## code present in src/processing.py
  
  os.chdir(path)

  if os.path.isdir('train') is False:
      os.makedirs('train')
      os.makedirs('train/dog')
      os.makedirs('train/cat')
      os.makedirs('valid/dog')
      os.makedirs('valid/cat')
      os.makedirs('test/dog')
      os.makedirs('test/cat')

      for i in random.sample(glob.glob('cat*'), 500):
          shutil.move(i, 'train/cat')      
      for i in random.sample(glob.glob('dog*'), 500):
          shutil.move(i, 'train/dog')
      for i in random.sample(glob.glob('cat*'), 100):
          shutil.move(i, 'valid/cat')        
      for i in random.sample(glob.glob('dog*'), 100):
          shutil.move(i, 'valid/dog')
      for i in random.sample(glob.glob('cat*'), 50):
          shutil.move(i, 'test/cat')      
      for i in random.sample(glob.glob('dog*'), 50):
          shutil.move(i, 'test/dog')

  os.chdir('../src/')
  ```
  
 * **Note:** We are not using all the images from the dataset, just using a few. If you want you can just change the loop iterations as per the count of images.
 
 * So we have `1000` train images `500` for dogs and `500` for cats. `200` validation images(`100` dogs, `100` cats) and `100` test images(`50` dogs and `50` cats).
 
## Processing The Data

* Using Keras' `ImageDataGenerator` class to create batches of data from the train, valid, and test directories.

  ```python
  ## code present in src/processing.py
  
  ## creating train-batches
  train_batches = ImageDataGenerator(
      preprocessing_function=tf.keras.applications.vgg16.preprocess_input
  ).flow_from_directory(directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)

  ## creating validation-batches
  valid_batches = ImageDataGenerator(
      preprocessing_function=tf.keras.applications.vgg16.preprocess_input
  ).flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)

  ## creating test-batches
  test_batches = ImageDataGenerator(
      preprocessing_function=tf.keras.applications.vgg16.preprocess_input
  ).flow_from_directory(directory=test_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10, shuffle=False)
  ```
  
* `ImageDataGenerator.flow_from_directory()` creates a `DirectoryIterator`, which generates batches of normalized tensor image data from the respective data directories.  

* To `ImageDataGenerator` for each of the data sets, we specify `preprocessing_function=tf.keras.applications.vgg16.preprocess_input`.

* We will be seesing VGG16 in later sections.

* To `flow_from_directory()`, we first specify the path for the data. We then specify the `target_size` of the images, which will resize all images to the specified size. The size we specify here is determined by the input size that the neural network expects.

* The `classes` parameter expects a `list` that contains the underlying class names, and lastly, we specify the `batch_size`.

* We also specify `shuffle=False` only for `test_batches`. That's because, later when we plot the evaluation results from the model to a confusion matrix, we'll need to able to access the unshuffled labels for the test set. By default, the data sets are shuffled.

