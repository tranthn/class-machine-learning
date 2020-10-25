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
# w <- w + eta * sum [(d - o) * xi]
#
# gradient descent
#   = 2 * (o - d) x

##################################################

# helper to factorize the class column to 1 / 0 based on target_class
# we will be doing one vs all (other) classes for adaline
def one_vs_all(df, label, target_class = ''):
    df[label] = (df[label] == target_class).astype(int)
    return df

def activation(x, w, bias):
    # x = feature matrix, dimensions = n x d
    # w = weights matrix, dimensions = d x 1
    # z = output, dimensions = n x 1
    z = np.dot(x, w) + bias
    return z

def adaline(x = None, y = None, w = None, eta = 0.05, iterations = 10):
    bias = 0.005

    for i in range(iterations):
        # z, dimensions = n
        z = activation(x, w, bias)

        # print('\ny shape')
        # print(y.shape)
        # print(y)
        # print('\nz shape')
        # print(z.shape)
        # print(z)

        # y, dimensions = n 
        # diff, dimensions = n        
        diff = (y - z)
        error = (diff ** 2)

        # print('\nw shape')
        # print(w.shape)
        # print('\nx.T shape')
        # print(x.T.shape)
        # print(x.T)
        # print('\ndiff shape')
        # print(diff.shape)
        # print(diff)

        # diff, dimensions = n
        # transpose x to line up dimensions (d x n)
        # print('\neta')
        # print(eta)
        # print('\ninner')
        # print(eta * x.T.dot(diff))
        # print()
        # print('\nerror')
        # print(error.sum())
        bias += eta * error.sum()

        w = w + eta * x.T.dot(diff)
        
        # print('\nupdated w shape')
        # print(w.shape)

    return w

# wrapper to calculate and update weights for our model
#
# arguments
#   - df: dataframe, contains all columns
#   - label: class label
#   - eta: the learning rate
#   - iterations: number of runs to run gradient descent and update weights
#
# returns
#   - w: final weights set for model
def build(df = None, label = '', eta = 0.005, iterations = 5):
    # get classes and the k-value (# class options)
    class_column = df[[label]]
    df_byclass = df.groupby(by = [label], dropna = False)
    classes = list(df_byclass.groups.keys())
    k = len(classes)

    #### calculate adaline with samples and weights ####
    # x = df without class column
    x = df.copy().drop(columns = label).to_numpy()
    d = x.shape[1]

    # weight map
    w_map = {}

    # for k = 2, we don't have do one versus all processing
    # but we'll have to set to map structure for consistency
    if (len(classes) == 2):
         # set weights 2-d frame, with dimensions d x 1
        w = np.random.uniform(-.01, 0.01, d)

        # y, dimensions = n
        y = df[label]

        # get weights for this class combo
        w = adaline(x, y, w, eta, iterations)

        # 1 main set of weights for binary classification
        w_map['main'] = w

    # for k > 2, we will process our classes to do one versus all (remaining) classes
    elif (k > 2):
        for target_class in classes:
            # set weights 2-d frame, with dimensions d x 1
            w = np.random.uniform(-.01, 0.01, d)

            # convert dataframe to target_class versus remainder
            df_one_vs_all = one_vs_all(df.copy(), label, target_class)

            # y, dimensions = n
            y = df_one_vs_all[label]

            # get weights for this class combo
            w = adaline(x, y, w, eta, iterations)
            
            # map the weights matrix to its target class
            w_map[target_class] = w

    return w_map

# calculate class prediction given data and weights
#
# arguments
#   - x: data without class column
#   - w: weights representing our model
#
# returns
#   - class predictions flattened 1-d array
def predict(x, w):
    # check if activation function  >= 0
    #  - return 1
    #  - otherwise return 0
    out =  activation(x, w, 0.05)
    return np.where(out > 0, 1, 0).flatten()

# run weights with our test data
#
# arguments
#   - df: dataframe (with all columns)
#   - w: weights representing our model (for a given target class)
#   - label: class label
#
# returns
#   - returns accuracy of prediction with our given dataframe
def test(df, w, label):
    #### calculate adaline with samples and weights ####
    # x = df without class column
    n = df.shape[0]
    x = df.copy().drop(columns = label)
    y = df[label]

    ### accuracy map
    accuracy_map = {}

    # predict classes with given weight array
    predictions = predict(x, w)

    # convert predictions to class label
    comp = np.equal(y.to_numpy(), predictions)
    corr = sum(comp)
    return (corr / n)

# wrapper to run test with weight map
#
# arguments
#   - df: dataframe (with all columns)
#   - weight_map: target class to weight array mapping
#   - label: class label
#
# returns
#   - returns accuracy map for each potential class for our dataframe
def test_multi_class_helper(df, weight_map, label):
    n = df.shape[0]

    # get class options
    classes = weight_map.keys()

    #### calculate adaline with samples and weights ####
    # x = df without class column
    x = df.copy().drop(columns = label)
    y = df[label]

    ### accuracy map
    accuracy_map = {}

    # predict classes with given weight array
    for target in classes:
        w = weight_map[target]
        accuracy = test(df, w, label)
        accuracy_map[target] = accuracy
    
    return accuracy_map