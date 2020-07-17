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
    for i in range(50):
        ## the ~5% young individuals who did experienced side effects
        random_young = randint(13, 64)
        train_samples.append(random_young)      ## the age of the individual [13, 64]
        train_labels.append(1)                  ## 1 -- experienced side effects

        ## the ~5% old individuals who did not experienced side effects
        random_old = randint(65, 100)
        train_samples.append(random_old)        ## the age of the individual [65, 100]
        train_labels.append(0)                  ## 0 -- did not experienced side effects

    for i in range(1000):
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
