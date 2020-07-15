# Keras With Tensorflow

## Overview:

* The notes has been made from [Deeplizard-Course](https://deeplizard.com/learn/video/RznKVRTFkBY).

* Here, we will learn how to use Keras, a neural network API written in Python! Each topic focuses on a specific concept and shows how the full implementation is done in code using Keras and Python.

    ![keras](https://user-images.githubusercontent.com/33928040/87566184-42911600-c6e0-11ea-95f3-4adc833beb56.png)

* We'll be starting from the basics by learning how to organize and preprocess data, and then we'll move on to building and training artificial neural networks.

* Some of the networks we'll build from scratch, and others will be pre-trained state-of-the-art models that we'll fine-tune to our data. After getting comfortable with building and training networks, we'll learn how to deploy our models using both front-end and back-end deployment techniques.

## Prerequisites:

* From a knowledge standpoint, many deep learning concepts will be explained and covered as we progress through the topics, however, if you are brand new to deep learning, then it is recommended that you start with a Deep Learning Fundamentals course first.

* In regards to coding prerequisites, just some basic coding skills and Python experience are needed!

## Keras:

* Keras was originally created by **François Chollet**. 

* Historically, Keras was a high-level API that sat on top of one of three lower level neural network APIs and acted as a wrapper to to these lower level libraries. These libraries were referred to as Keras backend engines.

* You could choose TensorFlow, Theano, or CNTK as the backend engine you’d like to work with.
    
    * TensorFlow
    * Theano
    * CNTK
    
* Ultimately, TensorFlow became the most popular backend engine for Keras.

* Later, Keras became integrated with the TensorFlow library and now comes completely packaged with it.    

    ![tensorflow](https://user-images.githubusercontent.com/33928040/87566665-f98d9180-c6e0-11ea-9574-6531eb64482d.png)

* Now, when you install TensorFlow, you also automatically get Keras, as it is now part of the TensorFlow library.

## How To Install Keras:

* Since Keras now comes packaged with TensorFlow, we need to install TensorFlow with the command:
  
  `pip install tensorflow`
  
* That's it. You can confim the installation by executing `import tensorflow` in your Python console.

## Hardware Requirements
* The only hardware requirement is having a NVIDIA GPU card with CUDA Compute Capability.

* Check the [TensorFlow website](https://www.tensorflow.org/install/gpu#hardware_requirements) for currently supported versions.

* Going forward, there are different instructions depending on if you’re running your code from a Windows environment or Linux environment. We’ll mostly go into depth on the Windows side, but first let’s touch on Linux.

### Linux Setup

* To simplify installation and avoid library conflicts, TensorFlow recommends using a TensorFlow Docker image with GPU support, as this setup only requires the NVIDIA GPU drivers to be installed.

* TensorFlow has a [guide](https://www.tensorflow.org/install/docker) with all the corresponding steps to get this set up.

### Windows Setup

* For Windows, the process is a bit more involved, you can check this [link](https://deeplizard.com/learn/video/IubEtS2JAiY) (scroll down till you find the Window Setup).

