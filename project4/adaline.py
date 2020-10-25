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
#   = 2 * (o - d) x

##################################################

# helper to factorize the class column to 1 / 0 based on target_class
# we will be doing one vs all (other) classes for adaline
def one_vs_all(df, label, target_class = ''):
    df[label] = (df[label] == target_class).astype(int)
    return df

def activation(x, w):
    # x = feature matrix, dimensions = n x d
    # w = weights matrix, dimensions = d x 1
    # z = output, dimensions = n x 1
    z = np.dot(x, w).flatten()
    return z

def adaline(x = None, y = None, w = None, eta = 0.05, target_class = '', iterations = 10):
    for i in range(iterations):
        # z, dimensions = n
        z = activation(x, w)
        print('\nz')
        print(z)
        print()

        # y, dimensions = n 
        # diff, dimensions = n
        diff = (y.to_numpy() - z)
        error = (diff ** 2)
        print('\ndiff')
        print(diff)
        print()

        print('\nx.T')
        print(x.T)
        print()

        # diff, dimensions = n
        # transpose x to line up dimensions (d x n)
        w = w + eta * 2 * np.dot(x.T, diff)

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
    x = df.copy().drop(columns = label)
    d = x.shape[1]

    # weight map
    w_map = {}

    for target_class in classes:
        # set weights 2-d frame, with dimensions d x 1
        w = np.random.uniform(-.01, 0.01, (d, 1))

        # convert dataframe to target_class versus remainder
        df_one_vs_all = one_vs_all(df.copy(), label, target_class)

        # classes, one-hot encoded here so that main dataframe class is left alone
        # y, dimensions = n
        y = df_one_vs_all[label]
        print('\ny')
        print(y)

        # get weights for this class combo
        w = adaline(x, y, w, class_column, eta, iterations)
        
        # map the weights matrix to its target class
        w_map[target_class] = w

    return w_map

def predict(x, w):
    # check if activation function  >= 0
    #  - return 1
    #  - otherwise return 0
    out =  activation(x, w)
    return np.where(out > 0, 1, 0)

def test(df, weight_map, label):
    n = df.shape[0]

    # get class options
    classes = weight_map.keys()

    #### calculate adaline with samples and weights ####
    # x = df without class column
    x = df.copy().drop(columns = label)

    ### accuracy map
    accuracy_map = {}

    # predict classes with given weight array
    for target in classes:
        w = weight_map[target]
        predictions = predict(x, w)

        # convert predictions to class label
        comp = np.equal(classes.to_numpy(), predictions)
        corr = sum(comp)
        print('target class', target)
        print('correct\t', corr)
        print('total\t', n)
        accuracy_map[target] = (corr / n)