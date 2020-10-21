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
#   - w: array of floats
#   - x: array of floats
#
# returns
#   - returns array with exponential value applied over
def softmax(x, w):
    # x = feature matrix, dimensions = n x d
    # w = weights matrix, dimensions = d x k
    # pick axis = 1 for dot product along the column (k)
    # z = output, dimensions = n x k
    z = x.dot(w)
    print('\nz')
    print(z)

    # denom dimensions = n
    denom = np.sum(np.exp(z), axis = 1).values
    print('\ndenom')
    print(denom)
    print()

    sfm = np.exp(z).T / denom
    print(sfm)
    return sfm.T

def logistic_multi(df = None, label = ''):
    n = df.shape[0]

    # get classes and the k-value (# class options)
    df_byclass = df.groupby(by = [label], dropna = False)
    classes = list(df_byclass.groups.keys())
    k = len(classes)
    k1 = k - 1 # for convenience

    #### calculate softmax with samples and weights ####
    # x = df without class column
    x = df.copy().drop(columns = label)
    d = x.shape[1]

    print('n = ', n)
    print('d = ', d)
    print('k = ', k)

    # set weights 2-d frame, with dimensions d x k
    w = np.random.uniform(-.01, 0.01, (d, k))
    print(w)

    # classes, one-hot encoded here so that main dataframe class is left alone
    # y, dimensions = n x k
    y = pd.get_dummies(df[[label]], columns = [label])
    probabilities = softmax(x, w)
 
    #### calculate weight changes ####
    # (real-value r_t - yi) times x
    # x, dimensions = n x d
    print('\n=========================')
    print(x.values)
    print()
    print(y) # dimensions = n x k
    print('\nprobs')
    print(probabilities) # dimensions = n x k
    diff = y.values - probabilities.values
    print('\ndiff')
    print(diff)

    # BROKEN IDK CHECK LATER
    w = w + x.dot(y - probabilities)

    return w