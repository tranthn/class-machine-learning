#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import math

## adaline logic
# x = input, w = weights
# z = w * x = sum(w_i * x_i)
# identity function [ o = z ] be the activation function
#
# squared error E = (d - o)^2
#
# weight update
# eta = learning rate, d = desired (real) output
# w <- w + eta (d - o) x
#
# gradient descent
#   = 2(o - d) x

##################################################

def activation(x, w):
    # x = feature matrix, dimensions = n x d
    # w = weights matrix, dimensions = d x k
    # pick axis = 1 for dot product along the column (k)
    # z = output, dimensions = n x k
    z = x.dot(w)
    return z

def adaline(df = None, label = '', iterations = 10):
    n = df.shape[0]

    # get classes and the k-value (# class options)
    df_byclass = df.groupby(by = [label], dropna = False)
    classes = list(df_byclass.groups.keys())
    k = len(classes)
    k1 = k - 1 # for convenience

    #### calculate adaline with samples and weights ####
    # x = df without class column
    x = df.copy().drop(columns = label)
    d = x.shape[1]

    print('n = ', n)
    print('d = ', d)
    print('k = ', k)

    # set weights 2-d frame, with dimensions d x k
    w = np.random.uniform(-.01, 0.01, k + 1)
    print('\nweights')
    print(w)

    # classes, one-hot encoded here so that main dataframe class is left alone
    y, levels = pd.factorize(df[label])
    # y = pd.get_dummies(df[[label]], columns = [label])

    # learning rate / eta, hard-coded here for now
    eta = 0.05

    for i in range(iterations):
        # z, dimensions = n x k
        z = activation(x, w)

        # y, dimensions = n (with k factors for the classes)
        diff = (y - z)
        error = (diff ** 2)

        # x, dimensions = n * d - NEED TO FIX [todo]
        w = w + eta * (np.dot(x, diff))

def predict(x, w):
    # check if activation function  >= 0
    #  - return 1
    #  - otherwise return 0
    out =  activation(x, w)
    return out

def test(df, w, label):
    n = df.shape[0]

    # get classes and the k-value (# class options)
    df_byclass = df.groupby(by = [label], dropna = False)
    classes = df[label]

    #### calculate adaline with samples and weights ####
    # x = df without class column
    x = df.copy().drop(columns = label)

    # classes, one-hot encoded here so that main dataframe class is left alone
    y, levels = pd.factorize(df[[label]])

    # predict classes with given weight array
    predictions = predict(x, w)

    # convert predictions to class label
    comp = np.equal(classes.to_numpy(), predictions.to_numpy())