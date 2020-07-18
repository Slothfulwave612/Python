# Tutorial 01

## Contents

* [Data Processing For Neural Network Training](#data-processing-for-neural-network-training)
  * [Samples and Labels](#samples-and-labels)
  * [Expected Data Format](#expected-data-fromat)
  * [Process Data in Code](#process-data-in-code)
  * [Data Creation](#data-creation)
  * [Data Processing](#data-processing)
  
* [ANN With TensorFlow's Keras API](#ann-with-tensorflows-keras-api)
  * [Code Setup](#code-setup)
  * [Building a Sequential Model](#building-a-sequential-model)
  * [Layers in The Model](#layers-in-the-model)
  
* [Training ANN](#training-ann)  
  * [Compiling The Model](#compiling-the-model)
  * [Training The Model](#training-the-model)
  
* [Building A Validation Set](#building-a-validation-set)  
  * [What is a Validation Set?](#what-is-a-validation-set)
  * [Creating a Validation Set](#creating-a-validation-set)
  * [Interpret Validation Metrics](#interpret-validation-metrics)

* **Note**: You can use src/run_script.py to run each model that is being discussed in this tutorial, for exploration part you can go to notebooks/explore.ipynb

## Data Processing For Neural Network Training

* In this section, we’ll see how to process numerical data that we’ll later use to train our very first artificial neural network.

### Samples and Labels

* To train any neural network in a supervised learning task, we first need a data set of samples and the corresponding labels for those samples.

* When referring to *samples*, we're just referring to the underlying data set, where each individual item or data point within that set is called a *sample*. 

* *Labels* are the corresponding labels for the samples(or the output of the sample).

* If we were training a model on images of cats and dogs, then the label for each of the images would either be “cat” or “dog.”

* **Note:** In Deep Learning, *samples* are also commonly referred to as *input data* or *inputs*, and *labels* are also commonly referred to as *target data* or *targets*.

### Expected Data Fromat

* When preparing data, we first need to understand the format that the data need to be in for the end goal we have in mind. 

* In our case, we want our data to be in a format that we can pass to a neural network model.

* The first model we'll build will be a **Sequential model** from the Keras API integrated within TensorFlow. We'll discuss the details of this type of model in that later sections, but for now, we just need to understand the type of data that is expected by a *Sequential model*.

* The Sequential model receives data during training, which occurs when we call the `fit()` function on the model. Let's now check the type of data this function expects.

* As per the [documentation](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit), the input data `x` need to be in one of the following data types:
  
  * A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
  * A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs).
  * A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
  * A tf.data dataset. Should return a tuple of either (inputs, targets) or (inputs, targets, sample_weights).
  * A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights).
  
* The corresponding labels `y` for the data are expected to be formatted similarly.  

* **Note:** If `x` is a dataset, generator, or `keras.utils.Sequence` instance, `y` should not be specified(since labels will be obtained from x).

* Aside from formatting the data to make it meet the format required by the model. We need to process the data as well.

* We transform our data in such a way that it may make it easier, faster or more efficient for the network to learn from.

* For that we can use *normalization* or *standardization* techniques.

### Process Data in Code

* Data processing for deep learning will vary greatly depending on the type of data we're working with and the type of task we'll be using the network for.

* In this tutorial we will start out with a very simple classification task using a simple numerical dataset.

* Let's import libraries we need:
  
  ```python
  ## note these modules may/may not be imported in the same file
  
  import numpy as np
  from random import randint
  from sklearn.utlis import shuffle
  from sklearn.preprocessing import MinMaxScaler
  ```
  
* We will now create two empty list. One will hold the *input data*, and the other will hold the *target data*.
  
  ```python
  train_samples = []     ## input data
  train_labels = []      ## target data
  ```
  
### Data Creation

* For this tutorial we are going to create our own dataset.

* **Motivation For This Data:**
  
  * Let’s suppose that an experimental drug was tested on individuals ranging from age `13` to `100` in a clinical trial. The trial had `2100` participants. Half of the participants were under `65` years old, and the other half was `65` years of age or older.
  
  * The trial showed that around `95%` of patients `65` or older experienced side effects from the drug, and around `95%` of patients under `65` experienced no side effects, generally showing that elderly individuals were more likely to experience side effects.

  * Ultimately, we want to build a model to tell us whether or not a patient will experience side effects solely based on the patient's age. The judgement of the model will be based on the training data.


* The block of code below shows how to generate this dummy data.
  
  ```python
  ## code present in src/create_data.py
  
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
  ```
  
  * This code creates `2100` samples and stores the age of the individuals in the `train_samples` list and stores whether or not the individuals experienced side effects in the `train_labels` list.
  
* Data in both the list looks like this:
  ```python
  ## code present in notebooks/explore.ipynb
  
  [ln]: train_samples[:10], train_labels[:10]
  [op]: ([55, 95, 25, 87, 50, 93, 46, 82, 51, 88], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
  ```

### Data Processing

* We now convert both lists into numpy arrays due to what we discussed the `fit()` function expects, and we then shuffle the arrays to remove any order that was imposed on the data during the creation process.

 ```python
 ## code present in src/processing.py
 
 ## convert list to numpy array
 train_samples = np.array(train_samples)
 train_labels = np.array(train_labels)

 ## shuffle the data
 train_samples, train_labels = shuffle(train_samples, train_labels)
 ```

* Now we will scale down our sample values from `13` to `100` to be on a scale from `0` to `1`.

  ```python
  ## code present in src/processing.py
  
  ## MinMaxScalar object
  scaler = MinMaxScaler(feature_range=(0, 1))

  ## scale train_samples
  scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))
  ```
  
* We reshape the data as a technical requirement just since the `fit_transform()` function doesn’t accept 1D data by default.  

* *Scaling* basically helps to normalise the data within a particular range. Sometimes, it also helps in speeding up the calculations in an algorithm.

* Here is how `scaled_train_samples` looks like after normalization.
  
  ```
  ## code present in notebooks/explore.ipynb
  
  [ln]: type(scaled_train_samples), type(train_labels)
  [op]: (numpy.ndarray, numpy.ndarray)
  
  [ln]: scaled_train_samples
  [op]: array([[1.        ],
       [0.32183908],
       [0.49425287],
       ...,
       [0.59770115],
       [0.55172414],
       [0.1954023 ]])
       
  [ln]: train_labels
  [op]: array([1, 0, 0, ..., 1, 0, 0])
  ```
  
  ## ANN With TensorFlow's Keras API
  
  * Now we will create a simple *Artificial Neural Network(ANN)* using a `Sequential` model from Keras API integrated with TensorFlow.
  
  ### Code Setup
  
  * These are the modules we are making use of:
    
    ```python
    ## note these modules may/may not be imported in the same file
    
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Activation, Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.metrics import categorical_crossentropy
    ```
    
* If you are working with a GPU, the following script will help you tell wheter GPU is working with tensorflow or not.
   
   ```python
   ## code present in src/train.py
   
   ## to see the physical devices
   physical_devices = tf.config.experimental.list_physical_devices('GPU')

   ## will display number of gpu integrated with tensorflow
   print(f'Number of GPU availabe: {len(physical_devices)}')

   ## to enable memory growth for a PhysicalDevice
   tf.config.experimental.set_memory_growth(physical_devices[0], True)
   ```

* `set_memory_growth()` attempts to allocate only as much GPU memory as needed at a given time, and continues to allocate more when needed. If this is not enabled, then we may end up running into the error below when we train the model later:

   `Blas GEMM launch failed`

* Output will be:
  
  ```python
  ## code present in notebooks/explore.ipynb
  
  [ln]: test_gpu()
  [op]: Number of GPU availabe: 1
  ```
  
### Building a Sequential Model

* Let's now create our first model:
  
  ```python
  ## code present in src/train.py
  
  ## Sequential Model
  model = Sequential(layers=[
      Dense(units=16, input_shape=(1,), activation='relu'),
      Dense(units=32, activation='relu'),       Dense(units=1, activation='sigmoid')
  ])
  ```
  
* `model` is an instance of a `Sequential` object. A `tf.keras.Sequential` model is a linear stack of layers. It accepts a list, and each element in the list should be a layer.

* As you can see, we have passed a list of layers to the `Sequential` constructor.

### Layers in The Model

* **First Hidden Layer:**
 
  * Our first layer is a `Dense` layer. This type of layer is our standard fully-connected or densely-connected neural network layer. 
  
  * The first required parameter that the `Dense` layer expects is the number of neurons or `units` the layer has, and we’re arbitrarily setting this to `16`.
  
  * Additionally, the model needs to know the shape of the input data. For this reason, we specify the shape of the input data in the first hidden layer in the model (and only this layer). The parameter called `input_shape`.
  
  * The `input_shape` parameter expects a tuple of integers that matches the shape of the input data, so we correspondingly specify `(1,)` as the `input_shape` of our one-dimensional data.
  
  * Lastly, an optional parameter that we’ll set for the `Dense` layer is the `activation` function to use after this layer. We’ll use the popular choice of `relu`. Note, if you don’t explicitly set an activation function, then Keras will use the `linear` activation function.
  
* **Second Hidden Layer:**
 
  * Our next layer will also be a `Dense` layer, and this one will have `32` nodes. 
  
  * The choice of how many neurons this node has is also arbitrary, as the idea is to create a simple model, and then test and experiment with it. 
  
  * If we notice that it is insufficient, then at that time, we can troubleshoot the issue and begin experimenting with changing parameters, like number of layers, nodes, etc.

  * This `Dense` layer will also use `relu` as its activation function.

* **Output Layer:**
  
  * Lastly, we specify the output layer. This layer is also a `Dense` layer, and it will have `1` neurons.
  
  * It will either output a `0` or `1`.
  
  * This time, the activation function we’ll use is `sigmoid`.

* In order to see a quick visualization of our model layers, we can use `model.summary()`.

  ```
  ## code is present in notebooks/explore.ipynb
  
  [ln]: model.summary()
  [op]: Model: "sequential"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        dense (Dense)                (None, 16)                32        
        _________________________________________________________________
        dense_1 (Dense)              (None, 32)                544       
        _________________________________________________________________
        dense_2 (Dense)              (None, 1)                 33        
        =================================================================
        Total params: 609
        Trainable params: 609
        Non-trainable params: 0
  ```

## Training ANN

* In this section, we’ll demonstrate how to train an artificial neural network using the Keras API integrated within TensorFlow.

### Compiling the Model

* The first thing we need to do to get the model ready for training is call the `compile()` function on it.
  
  ```python
  ## code present in src/train.py
  
  ## compiling the model
  model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  ```
  
* This function configures the model for training and expects a number of parameters. First, we specify the `optimizer` `Adam`. `Adam` accepts an optional parameter `learning_rate`, which we’ll set to `0.0001`.  

* The next parameter we specify is `loss`. We’ll be using `binary_crossentropy`, given that our labels are either 0 or 1(binary).

* The last parameter we specify in `compile()` is `metrics`. This parameter expects a list of metrics that we’d like to be evaluated by the model during training and testing. We’ll set this to a list that contains the string `'accuracy'`.

### Training the Model

* Now that the model is compiled, we can train it using the `fit()` function.
  
  ```python
  ## code present in src/train.py
  
  ## training the model
  model.fit(x=scaled_train_samples, y=train_labels, batch_size=10, epochs=30, verbose=2)
  ```

* The first item that we pass in to the `fit()` function is the training set x, which will be our `scaled_train_samples`.

* The next parameter is the label for training set `y`, which will be our `train_labels`.

* We then have specified the `batch_size` to be `10` and `epochs` to be 30.
  
  * **Note:** That an epoch is a single pass of all the data to the network.
  
* Lastly we specify, `verbose=2`. This specify how much output we want on the console. Verbosity ranges from 0 to 2, hence here we are getting the max verbose level.

* The output of the `fit()` function looks like this:
  
  ```
  Epoch 1/30
  210/210 - 1s - loss: 0.7017 - accuracy: 0.4305
  Epoch 2/30
  210/210 - 1s - loss: 0.6818 - accuracy: 0.6538
  Epoch 3/30
  210/210 - 1s - loss: 0.6654 - accuracy: 0.6962
  Epoch 4/30
  210/210 - 1s - loss: 0.6458 - accuracy: 0.7319
  Epoch 5/30
  210/210 - 1s - loss: 0.6265 - accuracy: 0.7529
  Epoch 6/30
  210/210 - 1s - loss: 0.6055 - accuracy: 0.7781
  Epoch 7/30
  210/210 - 1s - loss: 0.5853 - accuracy: 0.8029
  Epoch 8/30
  210/210 - 1s - loss: 0.5651 - accuracy: 0.8214
  Epoch 9/30
  210/210 - 1s - loss: 0.5431 - accuracy: 0.8310
  Epoch 10/30
  210/210 - 1s - loss: 0.5205 - accuracy: 0.8424
  Epoch 11/30
  210/210 - 1s - loss: 0.4980 - accuracy: 0.8576
  Epoch 12/30
  210/210 - 1s - loss: 0.4759 - accuracy: 0.8700
  Epoch 13/30
  210/210 - 1s - loss: 0.4546 - accuracy: 0.8819
  Epoch 14/30
  210/210 - 1s - loss: 0.4345 - accuracy: 0.8843
  Epoch 15/30
  210/210 - 1s - loss: 0.4158 - accuracy: 0.8962
  Epoch 16/30
  210/210 - 1s - loss: 0.3988 - accuracy: 0.8986
  Epoch 17/30
  210/210 - 1s - loss: 0.3831 - accuracy: 0.9000
  Epoch 18/30
  210/210 - 1s - loss: 0.3689 - accuracy: 0.9067
  Epoch 19/30
  210/210 - 1s - loss: 0.3561 - accuracy: 0.9095
  Epoch 20/30
  210/210 - 1s - loss: 0.3448 - accuracy: 0.9124
  Epoch 21/30
  210/210 - 1s - loss: 0.3350 - accuracy: 0.9205
  Epoch 22/30
  210/210 - 1s - loss: 0.3262 - accuracy: 0.9205
  Epoch 23/30
  210/210 - 1s - loss: 0.3185 - accuracy: 0.9186
  Epoch 24/30
  210/210 - 1s - loss: 0.3118 - accuracy: 0.9219
  Epoch 25/30
  210/210 - 1s - loss: 0.3057 - accuracy: 0.9248
  Epoch 26/30
  210/210 - 1s - loss: 0.3004 - accuracy: 0.9248
  Epoch 27/30
  210/210 - 1s - loss: 0.2957 - accuracy: 0.9281
  Epoch 28/30
  210/210 - 1s - loss: 0.2916 - accuracy: 0.9281
  Epoch 29/30
  210/210 - 1s - loss: 0.2879 - accuracy: 0.9286
  Epoch 30/30
  210/210 - 1s - loss: 0.2843 - accuracy: 0.9314
  ```
  
* We can see corresponding output for each of the `30` epochs. Judging by the loss and accuracy, we can see that both metrics steadily improves over time with accuracy reaching almost 93% and loss steadily decreasing.

## Building A Validation Set

* In this section, we’ll demonstrate how to use TensorFlow's Keras API to create a validation set on-the-fly during training.

### What is a Validation Set?

* Recall that we previously built a training set on which we trained our model. With each epoch that our model is trained, the model will continue to learn the features and characteristics of the data in this training set.

* The hope is that later we can take this model, apply it to new data, and have the model accurately predict on data that it hasn’t seen before based solely on what it learned from the training set.

* Before training begins, we can choose to remove a portion of the training set and place it in a validation set. Then during training, the model will train only on the training set, and it will validate by evaluating the data in the validation set.

* During each epoch we will see not only the loss and accuracy results for the training set, but also for the validation set.

* This allows us to see how well the model is generalizing on data it wasn't trained on.

* This also helps us see whether or not the model is overfitting. 

* Overfitting occurs when the model learns the specifics of the training data and is unable to generalize well on data that it wasn't trained on.
  
   * One way to discover overfitting is when your train-accuracy is better than test-accuracy or in other words your test-accuracy is very poor than your train-accuracy, you can say there is some amount of overfitting.
   
### Creating a Validation Set   

* There are two ways to create a validation set:
  
  1. *Manually Create Validation Set*
  2. *Create Validation Set with Keras*

* **Manually Create Validation Set**:
  
  * The first way is to create a data structure to hold a validation set, and place data directly in that structure in the same nature we did for the training set.
  
  * This data structure should be a tuple `valid_set = (x_val, y_val)` of Numpy arrays or tensors, where `x_val` is a numpy array or tensor containing validation samples, and `y_val` is a numpy array or tensor containing validation labels.
  
  * When we call `model.fit()`, we would pass in the validation set in addition to the training set. We pass the validation set by specifying the `validation_data` parameter. For manual splitting of the data set look for `train_valid_split()` function in `src/processing.py` file.
  
  ```python
  ## code pesent in src/train.py
  
  model.fit(
     x=scaled_train_samples
   , y=train_labels
   , validation_data=valid_set
   , batch_size=10
   , epochs=30
   , verbose=2
  )
  ```
  
  * **Create Validation Set with Keras**:
  
  * There is another way to create a validation set, and it saves a step!
  
  * If we don’t already have a specified validation set created, then when we call `model.fit()`, we can set a value for the `validation_split` parameter. It expects a fractional number between `0` and `1`. Suppose that we set this parameter to `0.1`(meaning 10% of training data will go to validation data).
  
  ```python
  ## coed present in src/train.py
  
  model.fit(
      x=scaled_train_samples
    , y=train_labels
    , validation_split=0.1
    , batch_size=10
    , epochs=30
    , verbose=2
  )
  ```
  
  * **Note**: Note that the `fit()` function shuffles the data before each epoch by default. When specifying the `validation_split` parameter, however, the validation data is selected from the last samples in the x and y data before shuffling.
  
### Interpret Validation Metrics

* Now, regardless of which method we use to create validation data, when we call `model.fit()`, then in addition to `loss` and `accuracy` being displayed for each epoch, we will now also see `val_loss` and `val_acc` to track the loss and accuracy on the validation set.

* Output will be something like this(values can differ because of randomness in the data):
  
  ```
  Epoch 1/30
  189/189 - 0s - loss: 0.6914 - accuracy: 0.4720 - val_loss: 0.6850 - val_accuracy: 0.4667
  Epoch 2/30
  189/189 - 0s - loss: 0.6651 - accuracy: 0.6048 - val_loss: 0.6655 - val_accuracy: 0.5429
  Epoch 3/30
  189/189 - 0s - loss: 0.6442 - accuracy: 0.6545 - val_loss: 0.6499 - val_accuracy: 0.5714
  Epoch 4/30
  189/189 - 0s - loss: 0.6226 - accuracy: 0.7079 - val_loss: 0.6290 - val_accuracy: 0.6143
  Epoch 5/30
  189/189 - 0s - loss: 0.5993 - accuracy: 0.7455 - val_loss: 0.6101 - val_accuracy: 0.6667
  Epoch 6/30
  189/189 - 0s - loss: 0.5773 - accuracy: 0.7693 - val_loss: 0.5905 - val_accuracy: 0.7048
  Epoch 7/30
  189/189 - 0s - loss: 0.5548 - accuracy: 0.7799 - val_loss: 0.5687 - val_accuracy: 0.7333
  Epoch 8/30
  189/189 - 0s - loss: 0.5318 - accuracy: 0.8048 - val_loss: 0.5469 - val_accuracy: 0.7667
  Epoch 9/30
  189/189 - 0s - loss: 0.5085 - accuracy: 0.8212 - val_loss: 0.5236 - val_accuracy: 0.7762
  Epoch 10/30
  189/189 - 0s - loss: 0.4854 - accuracy: 0.8365 - val_loss: 0.5010 - val_accuracy: 0.7905
  Epoch 11/30
  189/189 - 0s - loss: 0.4630 - accuracy: 0.8471 - val_loss: 0.4778 - val_accuracy: 0.8238
  Epoch 12/30
  189/189 - 0s - loss: 0.4418 - accuracy: 0.8603 - val_loss: 0.4567 - val_accuracy: 0.8476
  Epoch 13/30
  189/189 - 0s - loss: 0.4217 - accuracy: 0.8725 - val_loss: 0.4372 - val_accuracy: 0.8524
  Epoch 14/30
  189/189 - 1s - loss: 0.4032 - accuracy: 0.8804 - val_loss: 0.4162 - val_accuracy: 0.8714
  Epoch 15/30
  189/189 - 0s - loss: 0.3858 - accuracy: 0.8878 - val_loss: 0.3977 - val_accuracy: 0.8810
  Epoch 16/30
  189/189 - 0s - loss: 0.3702 - accuracy: 0.9000 - val_loss: 0.3830 - val_accuracy: 0.8810
  Epoch 17/30
  189/189 - 0s - loss: 0.3564 - accuracy: 0.8989 - val_loss: 0.3661 - val_accuracy: 0.9143
  Epoch 18/30
  189/189 - 0s - loss: 0.3441 - accuracy: 0.9085 - val_loss: 0.3533 - val_accuracy: 0.9143
  Epoch 19/30
  189/189 - 0s - loss: 0.3334 - accuracy: 0.9122 - val_loss: 0.3422 - val_accuracy: 0.9286
  Epoch 20/30
  189/189 - 0s - loss: 0.3241 - accuracy: 0.9143 - val_loss: 0.3316 - val_accuracy: 0.9286
  Epoch 21/30
  189/189 - 0s - loss: 0.3158 - accuracy: 0.9206 - val_loss: 0.3236 - val_accuracy: 0.9286
  Epoch 22/30
  189/189 - 0s - loss: 0.3085 - accuracy: 0.9206 - val_loss: 0.3156 - val_accuracy: 0.9286
  Epoch 23/30
  189/189 - 0s - loss: 0.3023 - accuracy: 0.9206 - val_loss: 0.3082 - val_accuracy: 0.9286
  Epoch 24/30
  189/189 - 0s - loss: 0.2969 - accuracy: 0.9265 - val_loss: 0.3016 - val_accuracy: 0.9381
  Epoch 25/30
  189/189 - 0s - loss: 0.2923 - accuracy: 0.9265 - val_loss: 0.2954 - val_accuracy: 0.9381
  Epoch 26/30
  189/189 - 0s - loss: 0.2882 - accuracy: 0.9291 - val_loss: 0.2919 - val_accuracy: 0.9381
  Epoch 27/30
  189/189 - 0s - loss: 0.2846 - accuracy: 0.9291 - val_loss: 0.2857 - val_accuracy: 0.9476
  Epoch 28/30
  189/189 - 0s - loss: 0.2815 - accuracy: 0.9323 - val_loss: 0.2829 - val_accuracy: 0.9381
  Epoch 29/30
  189/189 - 0s - loss: 0.2789 - accuracy: 0.9312 - val_loss: 0.2789 - val_accuracy: 0.9476
  Epoch 30/30
  189/189 - 0s - loss: 0.2763 - accuracy: 0.9323 - val_loss: 0.2767 - val_accuracy: 0.9381
  ```
  
* We can now see not only how well our model is learning the features of the training data, but also how well the model is generalizing to new, unseen data from the validation set.   
