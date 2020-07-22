# Handling Missing Values

* **Note:** These notes are made from different youtube videos, github repos and medium articles.

## Content

* [Overview](#overview)

* [Types of Missing Values](#types-of-missing-values)
  * [Missing Completely At Random](#missing-completely-at-random)
  * [Missing At Random](#missing-at-random)
  * [Missing Not At Random](#missing-not-at-random)

* [Techniques of Dealing With Missing Data](#techniques-for-dealing-with-missing-data)
  * [Drop Missing Values](#drop-missing-values)
  * [Imputation](#imputation)
    * [Mean Median Mode Replacement](#mean-meadian-mode-replacement)

## Overview

* **Missing data** are defined as values that are not available and that would be meainingful if they are observed.

* Missing data can be anything from missing sequence, incomplete feature, data entry error etc. 

* Most dataset in the real world contains missing data. Before you can use data with missing fields, you need to transform those fields so they can be used for analysis and modelling. Understanding the data and the domain from which it comes is very important. 

* Having missing values in your data is not necessarily a setback but it is an opportunity to perform right feature engineering to guide the model to interpret the missing information right way.

* These missing values arise dur to many factors:
  
  * Sometimes due to the ways in which data is captured.
  
  * In some cases the values are not available for observation.
  
* Let's look at an example to see how a dataset can have missing values:
  
  * Let's say we want to analyse health of individuals, well for that we want to have a dataset.
  
  * So we went ahead and collected some data from different sources. Let's say we collected data from three different sources.
  
  * The data from *Source 01* has only height and mass of an individual and based on that the data has marked who is fit and who is unfit(basically a BMI).
  
  * The data from *Source 02* has some other factors like average calorie intake of an individual.
  
  * The data from *Source 03* has some more factors like whether an individual is doing exercise or not or what about the mental health of the individual etc etc.
  
  * So after collecting data from these three different source when we combine to make a full dataset, we will find missing values in our data as different sources are covering different parameters and it may be possible that an individual, let say A, has data in first two sources but not in the third because s/he was not surveyed by the third source.
  
  * So here, we can see how a dataset end up having missing values.

## Types of Missing Values

* There are three type of missing values we can found in a dataset:
  
  * *Missing Completely At Random* (MCAR)
  * *Missing At Random* (MAR)
  * *Missing Not At Random* (MNAR)
  
* **Note:** To inform all of our methods here we will go through the same example. Let's say in your town, you are trying to do a study: *on average how many overdue library books does each citizen have*. Now, we will be making different cases to see the three types of missing values.
  
### Missing Completely At Random

* When we say data are *missing completely at random*, we mean that the missingness has nothing to do with the observation being studied.

* Here in this case, the data goes missing at a completely consistent rate no matter what.

* Example:
  
  * In this first case, let's say you go to the library and in the computer you start finding values of how many overdue books people have.
  
  * Maybe someone has 1 overdue book, someone has 2, someone else has 5 but then you get to a very suspicious value *?*(it's missing). 
  
  * And you find out from the librarian that, about 5% of the time the librarians forget to type the values in, just human error, and this 5% is completely random, it does not depend on any other variables.
  
  * This is one such case of being missing completely at random.
  
### Missing At Random  
  
*  In missing at random, the data is missing at a certain rate but that rate depends on some other variables in the data.

* Example:

  * Let's pretend that you now don't have any access to the computer, and now you do a poll for your study. 

  * And let's say furthermore you are also interested in whether males and females have more overdue books, so you ask each person their gender and you also ask them what is the number of overdue books you currently have.

  * Furthermore, let's say that females are are applied to your poll at a 90% rate,so there is a 10% of missing data in a females. While for males, they're responding at a 70% rate so there's a much higher 30% missing data rate for males.

  * This is a perfect example of a case where the data is missing at random. Because although we have missing data overall the rate of missing data can be perfectly explained if we know certain other factor in this case it's the gender.

  * This is a simple case, but in more complicated cases you can have combinations of factors for example if you know the gender, their geographic location and also the family income then you can say something about the missing data rate.
    
### Missing Not At Random    

* When data are missing not at random, the missingness is related to the value itself.

* Example:
  
  * Now here, let's say you are making a a poll, where you are asking about the names of the individuals and number of overdue books they have.
  
  * So this is fundamentally different from the poll above because here you can easily identify the person so if someone tells you if I have 15 overdue books and you ask for their names and now you can identify who has 15 overdue books .
  
  * So, to guide this example let's say that if you have 0 overdue books then you are 95% likely to answer this poll. Let's say if you have one overdue book the percentage drops to 80% and for 2 books the percentage is 70% and so on.
  
  * This makes sense because if you have more overdue books you are less likely to tell me about it because it's kind of embrassing fact about your life ;).
  
  * So the general idea is the more overdue books you truly have the less likely you are going to tell me about that number and the more likely I am to have a missing value.

## Techniques of Dealing With Missing Data

* There are few techniques which can help you deal with missing values in your dataset:
  
  * *Drop Missing Values/Columns/Rows*
  
  * *Imputation*
  
### Drop Missing Values

* The simplest way to go forward is to drop the columns/rows for which the data is not available.

* But before making the decision to drop missing value/rows/columns, you have to consider a few things.

* If we drop the rows our total number of data points to train our model will go down which can reduce the model performance.
  
  * Do this only if you have large number of training examples and the rows with missing data are not very high in number.
  
* Dropping the column altogether will remove a feature from our model i.e the model predictions will be independent of that column/s.  
  
  * Drop columns if the data is missing for more than 60% observations but only if that variable is insignificant.
  
* In general dropping data is not a good approach in most cases since you loose a lot of potentially useful information, and we have better techniques to deal with missing data

### Imputation

* A slightly better approach towards handling missing data is **imputation**. 

* Imputation means to replace or fill the missing data with some value.

* Some of the imputation techniques that we will talk about are:
  
  * *Mean/Median/Mode Replacement*
  * *Random Sample Imputation*
  * *Capturing NaN values with a new feature* 
  * *End of Distribution Imputation*
  * *Arbitrary Imputation*
  * *Frequency Categories Imputation*
  
* **Note:** You can see the implementation of these technique in `explore.ipynb` and the functions are defined in `imputation.py`. Dataset used is `titanic`.
  
#### Mean Meadian Mode Replacement

* Mean/median/mode imputation has the assumption that the data are missing completely at random(MCAR). 

* **Which is better, replacement by mean and replacement by median?**:
  
  * It always depends on your data and your task.
  
  * If there is a dataset that have great outliers, go for median. E.x.: 99% of household income is below 100, and 1% is above 500, else you can use mean.

* **Advantages:**
  
  * Easy to implement.
  
  * Robust to outliers.
  
  * A faster way of filling the NaN values.
  
* **Disadvantages:**  
  
  * Changes or distort the original variance/standard-deviation.
  
  * Impacts correlation.
  
#### Random Sample Imputation

* In random sample imputation, we take random observation from the dataset and we use it to fill the NaN values.

* It also assumes that the data are missing completely at random(MCAR).

* **Advantages:**
  
  * Easy to implement.
  
  * Less distortion in variance/standard-deviation.
  
* **Disadvantage:**  
  
  * In every situation randomness won't work.
