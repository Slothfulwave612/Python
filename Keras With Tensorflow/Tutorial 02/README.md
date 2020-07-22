## Convolutional Neural Network

## Contents

* [Overview](#overview)

* [Directory Tree](#directory-tree)

* [Organize The Data](#organize-the-data)

* [Processing The Data](#processing-the-data)

* [Visualize The Data](#visualize-the-data)

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

* **Note:** In the case where you do not know the labels for the test data, you will need to modify the `test_batches` variable. Specifically, the change will be to set the parameters `classes = None` and `class_mode = None` in `flow_from_directory()`. 

## Visualize The Data 

* We now call `next(train_batches)` to generate a batch of images and labels from the training set. 

* **Note:** That the size of this batch is determined by the batch_size we set when we created train_batches. 

  ```python
  ## code present in notebooks/explore.ipynb
  
  ## for plotting images
  imgs, labels = next(train_batches)
  plot_images(imgs)
  print(labels)
  ```
  
* Now here is how plotting is done.
  
  ```python
  ## code present in src/processing.py
  
  ## create subplots
  _, axes = plt.subplots(nrows=1, ncols=10, figsize=(20,20))

  axes = axes.flatten()

  ## plotting image
  for img, ax in zip(image_arr, axes):
      ax.imshow(img)
      ax.axis('off')

  plt.tight_layout()
  plt.show()
  ```

* This is what the first processed random batch from the training set looks like.
  
  ![scr](https://user-images.githubusercontent.com/33928040/88193342-2f8bc200-cc5b-11ea-800e-6d4d4a09e119.png)
  
* Notice that the color appears to be distorted. This has to do with the VGG16 processing we applied to the data sets.

* **Note:** Dogs are represented with the `one-hot encoding` of `[0,1]`, and cats are represented by `[1,0]`. 

### Building A Simple CNN

* To build the CNN, we’ll use a Keras `Sequential` model.

  ```python
  ## create model
  model = Sequential(
      [
          Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
          MaxPool2D(pool_size=(2, 2), strides=2),
          Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
          MaxPool2D(pool_size=(2, 2), strides=2),
          Flatten(),
          Dense(units=1, activation='sigmoid')
      ]
  )
  ```
  
* The first layer in the model is a 2-dimensional convolutional layer. This layer will have `32` output filters each with a kernel size of `3x3`, and we’ll use the `relu` activation function.   

* **Note:** Note that the choice for the number of output filters specified is arbitrary, and the chosen kernel size of `3x3` is generally a very common size to use. 

* We enable zero-padding by specifying `padding = 'same'`.

* On the first layer only, we also specify the `input_shape`, which is the shape of our data. Our images are `224` pixels high and `224` pixels wide and have `3` color channels: RGB. This gives us an `input_shape` of `(224,224,3)`. 

* We then add a max pooling layer to pool and reduce the dimensionality of the data.

* We follow this by adding another convolutional layer with the exact specs as the earlier one, except for this second `Conv2D` layer has `64` filters. The choice of `64` here is again arbitrary, but the general choice of having more filters in later layers than in earlier ones is common. 

* This layer is again followed by the same type of `MaxPool2D` layer. 

* We then `Flatten` the output from the convolutional layer and pass it to a `Dense` layer. This `Dense` layer is the output layer of the network, and so it has 1 nodes, and we are using the softmax activation function.

* 
