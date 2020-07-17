# Tutorial 01

## Contents:

* [Data Processing For Neural Network Training](#data-processing-for-neural-network-training)
  * [Samples and Labels](#samples-and-labels)
  * [Expected Data Format](#expected-data-format)

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
