#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import math

## gradient descent pseudocode
# initialize weights by assigning random values in [-0.01, 0.01]
#   - d = array size, # features
# repeat
#   - from 0...d, set delta weights (weight changes) to all 0's
#   - from 1...N 
#       - set output o of model to 0 as well
#       - inner loop: j = 0...d
#           - set output of model to o = o + w_j * x_tj # go across all dimensions of vector to add up feature x weight
#       - y = sigmoid(o) # pass result through logistic function to predict class
#       - calculate gradient to determine new weight update, using (real value - y)
#   - apply weight update and go onto next iteration
# stop at convergence (which is?)

##################################################

# logistic regression for k > 2
#
# arguments
#   - z
#
# returns
#   - returns array with exponential value applied over
def softmax(z):
    # z = x * w, dimensions = n x k
    # denom dimensions = n, axis = 1 is along columns
    denom = np.sum(np.exp(z), axis = 1)
    sfm = np.exp(z).T / denom
    return sfm.T

def logistic_multi(x = None, w = None, k = 1, class_column = None, label = ''):
    n = x.shape[0] # number of samples
    d = x.shape[1] # number of features / columns

    # classes, one-hot encoded here so that main dataframe class is left alone
    # y, dimensions = n x k
    y = pd.get_dummies(class_column, columns = [label])

    print('\nweights (w)')
    print(w)

    # x = feature matrix, dimensions = n x d
    # w = weights matrix, dimensions = d x k
    # pick axis = 1 for dot product along the column (k)
    # z = output, dimensions = n x k
    z = np.dot(x, w)
    probabilities = softmax(z)
 
    ### gradient
    # x, dimensions = n x d
    # y, dimensions = n x k
    # probabilities, dimensions = n x k
    # (real-value r_t - yi) times x
    gradient = (-1 / n) * np.dot(x.T, np.subtract(y, probabilities)) + w

    ### loss calculation 
    loss = -np.sum(y * np.log(probabilities)) + np.sum(w * w)

    return gradient, loss

def build(df = None, label = '', iterations = 5):
    n = df.shape[0]

    # get classes and the k-value (# class options)
    class_column = df[[label]]
    df_byclass = class_column.groupby(by = [label], dropna = False)
    classes = list(df_byclass.groups.keys())
    print('\nclasses')
    print(classes)
    k = len(classes)

    #### calculate softmax with samples and weights ####
    # x = df without class column
    x = df.copy().drop(columns = label)
    d = x.shape[1]

    print()
    print('n = ', n)
    print('d = ', d)
    print('k = ', k)
    
    # set weights 2-d frame, with dimensions d x k
    w = np.random.uniform(-.01, 0.01, (d, k))
    eta = 0.5
    for i in range(iterations):
        gradient, loss = logistic_multi(x, w, k, class_column, label)
        w = w - (eta * gradient)
        print('\nloss')
        print(loss)
        print()
    
    return w

def predict(x, w):
    # dimensions = n x k
    z = np.dot(x, w)

    # probabilities, dimensions = n x k
    probabilities = softmax(z)
    predictions = np.argmax(probabilities, axis = 1)
    return predictions

def test(df, w, label):
    n = df.shape[0]

    # get original class column for comparison
    # convert df into x and y frames for features and dummied classes
    classes = df[label]
    x = df.copy().drop(columns = label)
    y = pd.get_dummies(df[[label]], columns = [label])

    # predict classes with given weight array
    predictions = predict(x, w)
    print('\ntest predictions')

    # convert classes to factors, to use with predictions
    vals, levels = pd.factorize(classes)
    predictions_with_class = levels.take(predictions)
    comp = np.equal(classes.to_numpy(), predictions_with_class.to_numpy())
    corr = sum(comp)
    print('correct\t', corr)
    print('total\t', n)