#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import math

##################################################
class NeuralNet():
    def __init__(self, label = '', eta = 0.05, iterations = 1):
        self.bias = 0.05
        self.label = label
        self.eta = eta
        self.iterations = iterations

    # helper to factorize the class column to 1 / 0 based on target_class
    # we will be doing one vs all (other) classes for model
    def one_vs_all(self, df, target_class = ''):
        df[self.label] = (df[self.label] == target_class).astype(int)
        return df

    # compute dot product for dataframe and weights
    #
    # arguments:
    #   - x: dataframe without class column, dimensions = n x d
    #   - w: weights, dimensions = d x 1
    #
    # returns
    #   - z: dot product of x * w plus bias value
    def activation(self, x, w):
        # z = output, dimensions = n x 1
        z = np.dot(x, w) + self.bias
        return z

    # the main function for calculating gradient and loss
    #
    # arguments
    #   - x: dataframe without class column
    #   - y: class column from dataframe
    #   - w: weights, may be random weights or the updated weights passed by caller
    #
    # returns:
    #   - w: final weights after all iterations
    def update_weights(self, x = None, y = None, w = None):
        for i in range(self.iterations):
            # z, dimensions = n
            z = self.activation(x, w)

            # y, dimensions = n 
            # diff, dimensions = n        
            diff = (y - z)

            # sum of squared errors (or residuals)
            sse = (diff ** 2).sum()

            # [todo] get derivative of sse w.r.t to predicted
            #   - something like [2 (y - z) ]
            #   - if derivative is w.r.t to z, then maybe: [ - 2 (y - z) * d(predicted) / d(b) ]
            #
            # we want d(sse) == 0
            # d (derivative) of predicted / d (b) = d / d(b) * activation function output
            # perhaps bias should be renamed to step size?

            # stepsize = (summed output of [ d(sse) / d(b) ]) * learning rate eta
            # new b = old b - stepsize

            # diff, dimensions = n
            # transpose x to line up dimensions (d x n)
            self.bias += self.eta * diff.sum()
            w += self.eta * x.T.dot(diff)

        return w

    # wrapper to calculate and update weights for our model
    #
    # arguments
    #   - df: dataframe, contains all columns
    #
    # returns
    #   - w: final weights set for model
    def build(self, df = None):
        # get classes and the k-value (# class options)
        class_column = df[[self.label]]
        df_byclass = df.groupby(by = [self.label], dropna = False)
        classes = list(df_byclass.groups.keys())
        k = len(classes)

        #### calculate model with samples and weights ####
        # x = df without class column
        x = df.copy().drop(columns = self.label).to_numpy()
        d = x.shape[1]

        # weight map
        w_map = {}

        # for k = 2, we don't have do one versus all processing
        # but we'll have to set to map structure for consistency
        if (len(classes) == 2):
            w = np.random.uniform(-.01, 0.01, d)
            y = df[self.label] # y, dimensions = n

            # get weights for this class combo
            w = self.update_weights(x, y, w)

            # 1 main set of weights for binary classification
            w_map['main'] = w

        # for k > 2, we will process our classes to do one versus all (remaining) classes
        elif (k > 2):
            for target_class in classes:
                w = np.random.uniform(-.01, 0.01, d)

                # convert dataframe to target_class versus remainder
                df_one_vs_all = self.one_vs_all(df.copy(), target_class)
                y = df_one_vs_all[self.label]

                # get weights for this class combo
                w = self.update_weights(x, y, w)
                
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
    def predict(self, x, w):
        # check if activation function  >= 0
        #  - return 1
        #  - otherwise return 0
        out = self.activation(x, w)
        return np.where(out > 0, 1, 0).flatten()

    # run weights with our test data
    #
    # arguments
    #   - df: dataframe (with all columns)
    #   - w: weights representing our model (for a given target class)
    #
    # returns
    #   - returns accuracy of prediction with our given dataframe
    def test(self, df, w):
        #### calculate model with samples and weights ####
        # x = df without class column
        n = df.shape[0]
        x = df.copy().drop(columns = self.label)
        y = df[self.label]

        ### accuracy map
        accuracy_map = {}

        # predict classes with given weight array
        predictions = self.predict(x, w)

        # convert predictions to class label
        comp = np.equal(y.to_numpy(), predictions)
        corr = sum(comp)
        return (corr / n)

    # wrapper to run test with weight map
    #
    # arguments
    #   - df: dataframe (with all columns)
    #   - weight_map: target class to weight array mapping
    #
    # returns
    #   - returns accuracy map for each potential class for our dataframe
    def test_multi_class_helper(self, df, weight_map):
        n = df.shape[0]

        # get class options
        classes = weight_map.keys()

        #### calculate model with samples and weights ####
        # x = df without class column
        x = df.copy().drop(columns = self.label)
        y = df[self.label]

        ### accuracy map
        accuracy_map = {}

        # predict classes with given weight array
        for target in classes:
            w = weight_map[target]

            # convert dataframe to target_class versus remainder
            df_one_vs_all = self.one_vs_all(df.copy(), target)
            accuracy = self.test(df_one_vs_all, w)
            accuracy_map[target] = accuracy

            acc_str = '{:.2%}'.format(accuracy)
            print('target class\t{0}\t{1}'.format(target, acc_str))

        return accuracy_map