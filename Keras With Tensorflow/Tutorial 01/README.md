# Tutorial 01

## Contents:

* [Data Processing For Neural Network Training](#data-processing-for-neural-network-training)
  * [Samples and Labels](#samples-and-labels)
  * [Expected Data Format](#expected-data-fromat)
  * [Process Data in Code](#process-data-in-code)
  * [Data Creation](#data-creation)

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
  ## Code present in src/create_data.py
  
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
