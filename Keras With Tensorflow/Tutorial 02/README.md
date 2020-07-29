## Convolutional Neural Network

## Contents

* [Overview](#overview)

* [Directory Tree](#directory-tree)

* [Organize The Data](#organize-the-data)

* [Processing The Data](#processing-the-data)

* [Visualize The Data](#visualize-the-data)

* [Building A Simple CNN](#building-a-simple-cnn)

* [Training The Model](#training-the-model)

* [Testing and Evaluatiion](#testing-and-evaluation)

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
  
  ## changing the path
  os.chdir(path)

  ## creating split
  if os.path.isdir('train') is False:

      ## making required directories
      os.mkdir('train')
      os.mkdir('train/dog')
      os.mkdir('train/cat')
      os.mkdir('valid/dog')
      os.mkdir('valid/cat')
      os.mkdir('test/dog')
      os.mkdir('test/cat')

      ## moving 1000 random cat images to train/cat
      for i in random.sample(glob.glob('cat*'), 1000):
          shutil.move(i, 'train/cat')

      ## moving 1000 random dog images to train/dog
      for i in random.sample(glob.glob('dog*'), 1000):
          shutil.move(i, 'train/dog')

      ## moving 300 random cat images to valid/cat
      for i in random.sample(glob.glob('cat*'), 300):
          shutil.move(i, 'valid/cat')

      ## moving 300 random dog images to valid/dog
      for i in random.sample(glob.glob('dog*'), 300):
          shutil.move(i, 'valid/dog')

      ## moving 100 random cat images to test/cat
      for i in random.sample(glob.glob('cat*'), 100):
          shutil.move(i, 'test/cat')

      ## moving 100 random cat images to test/dog
      for i in random.sample(glob.glob('cat*'), 100):
          shutil.move(i, 'test/dog')

  os.chdir('..')
  ```
  
 * **Note:** We are not using all the images from the dataset, just using a few. If you want you can just change the loop iterations as per the count of images.
 
 * So we have `2000` train images `1000` for dogs and `100` for cats. `600` validation images(`300` dogs, `390` cats) and `200` test images(`100` dogs and `100` cats).
 
## Processing The Data

* Using Keras' `ImageDataGenerator` class to create batches of data from the train, valid, and test directories.

  ```python
  ## code present in src/processing.py
  
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
  ```
  
* `ImageDataGenerator.flow_from_directory()` creates a `DirectoryIterator`, which generates batches of normalized tensor image data from the respective data directories.  

* To `flow_from_directory()`, we first specify the path for the data. We then specify the `target_size` of the images, which will resize all images to the specified size. The size we specify here is determined by the input size that the neural network expects.

* The `classes` parameter expects a `list` that contains the underlying class names, and lastly, we specify the `batch_size`.

* **Note:** In the case where you do not know the labels for the test data, you will need to modify the `test_batches` variable. Specifically, the change will be to set the parameters `classes = None` and `class_mode = None` in `flow_from_directory()`. 

## Visualize The Data 

* We now call `next(train_batches)` to generate a batch of images and labels from the training set. 

* **Note:** That the size of this batch is determined by the batch_size we set when we created train_batches. 

  ```python
  ## code present in notebooks/explore.ipynb
  
  ## for plotting images
  imgs, labels = next(train_batches)
  plot_images(imgs, labels)
  ```
  
* Now here is how plotting is done.
  
  ```python
  ## code present in src/processing.py
  
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
  ```

* This is what the first processed random batch from the training set looks like.
  
  ![src](https://user-images.githubusercontent.com/33928040/88648252-71e25280-d0e4-11ea-8180-6061512bbc26.png)

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
          Dense(units=2, activation='ssoftmax')
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

* We then `Flatten` the output from the convolutional layer and pass it to a `Dense` layer. This `Dense` layer is the output layer of the network, and so it has 2 nodes, and we are using the `softmax` activation function.

* We can check out model summary by using `model.summary()`.

  ```
  Model: "sequential_3"
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #   
  =================================================================
  conv2d_6 (Conv2D)            (None, 224, 224, 32)      896       
  _________________________________________________________________
  max_pooling2d_6 (MaxPooling2 (None, 112, 112, 32)      0         
  _________________________________________________________________
  conv2d_7 (Conv2D)            (None, 112, 112, 64)      18496     
  _________________________________________________________________
  max_pooling2d_7 (MaxPooling2 (None, 56, 56, 64)        0         
  _________________________________________________________________
  flatten_3 (Flatten)          (None, 200704)            0         
  _________________________________________________________________
  dense_3 (Dense)              (None, 2)                 401410    
  =================================================================
  Total params: 420,802
  Trainable params: 420,802
  Non-trainable params: 0
  ```
  
* Now that the model is built, we compile the model using the `Adam` optimizer with a learning rate of `0.0001`, a loss of `binary_crossentropy`, and we’ll look at `accuracy` as our performance `metric`.

### Training The Model

* Using `model.fit()` we will be training the model.

  ```python
  ## training the model
  model.fit(
      x = train_batches, 
      steps_per_epoch = len(train_batches),
      validation_data = valid_batches,
      validation_steps = len(valid_batches),
      epochs = 10,
      verbose = 2
  )
  ```
  
* We need to specify `steps_per_epoch` to indicate how many batches of samples from our training set should be passed to the model before declaring one epoch complete. Similarly, we specify `validation_steps` in the same fashion but with using `valid_batches`.

* We’re specifying `10` as the number of `epochs` we’d like to run, and setting the `verbose` parameter to 2, which just specifies the verbosity of the log output printed to the console during training.

* When we run this line of code, we can see the output of the model over `30` epochs.

  ```
  Epoch 1/10
  200/200 - 8s - loss: 0.6709 - accuracy: 0.5835 - val_loss: 0.6211 - val_accuracy: 0.6333
  Epoch 2/10
  200/200 - 7s - loss: 0.5895 - accuracy: 0.6830 - val_loss: 0.7808 - val_accuracy: 0.5417
  Epoch 3/10
  200/200 - 8s - loss: 0.5211 - accuracy: 0.7520 - val_loss: 0.5931 - val_accuracy: 0.6767
  Epoch 4/10
  200/200 - 7s - loss: 0.4503 - accuracy: 0.8140 - val_loss: 0.5810 - val_accuracy: 0.6983
  Epoch 5/10
  200/200 - 7s - loss: 0.3703 - accuracy: 0.8650 - val_loss: 0.5954 - val_accuracy: 0.7167
  Epoch 6/10
  200/200 - 7s - loss: 0.3178 - accuracy: 0.8785 - val_loss: 0.5860 - val_accuracy: 0.6933
  Epoch 7/10
  200/200 - 8s - loss: 0.2561 - accuracy: 0.9300 - val_loss: 0.6176 - val_accuracy: 0.6983
  Epoch 8/10
  200/200 - 8s - loss: 0.2116 - accuracy: 0.9410 - val_loss: 0.6321 - val_accuracy: 0.6950
  Epoch 9/10
  200/200 - 7s - loss: 0.1743 - accuracy: 0.9585 - val_loss: 0.7749 - val_accuracy: 0.6633
  Epoch 10/10
  200/200 - 7s - loss: 0.1437 - accuracy: 0.9750 - val_loss: 0.6838 - val_accuracy: 0.6750
  ```
  
* Our model is overfitting, let's see how it'll perform in test set.  

### Testing and Evaluation

* 
