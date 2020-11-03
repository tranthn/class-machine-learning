#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import math

##################################################
class NeuralNet():
    def __init__(self, df = None, label = '', eta = 0.05, iterations = 1,
                    num_hidden_nodes = 1, num_hidden_layers = 1, num_output_nodes = 1):

        self.df = df
        self.label = label
        self.eta = eta
        self.iterations = iterations

        # boundary sizes for our network structure
        self.num_hidden_nodes = num_hidden_nodes
        self.num_hidden_layers = num_hidden_layers
        self.num_output_nodes = num_output_nodes

        # network base structure
        self.network = {
            'input': None,
            'layers': [],
            'outputs': []
        }

        self.initialize()

    def pretty_print(self):
        print('NEURAL NETWORK')
        print('# layers\t', self.num_hidden_layers)
        print('# hidden nodes\t', self.num_hidden_nodes)
        print('# output nodes\t', self.num_output_nodes)
        print()

        print('LAYERS')
        print('---------')
        for l in self.network['layers']:
            print('layer')
            for w in l:
                print(w)
            print()

        print('\nOUTPUTS')
        print('---------')
        for o in self.network['outputs']:
            print(o)

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def sigmoid_derivative(self, input):
        o = sigmoid(input)
        dS = o * (1 - o)

    """
        the main function for calculating gradient and loss
        
        arguments
        - x: dataframe without class column
        - y: class column from dataframe
        - w: weights, may be random weights or the updated weights passed by caller
        
        returns:
        - w: final weights after all iterations
    """
    def update_weights(self, x = None, y = None, w = None):
        for i in range(self.iterations):
            # z, dimensions = n
            z = self.activation(x, w)

            # y, dimensions = n 
            # diff, dimensions = n        
            diff = (y - z)

            # sum of squared errors (or residuals)
            sse = (diff ** 2).sum()

            # [todo] get derivative of sse (w.r.t) predicted
            #   - something like [2 (y - z) ]
            #   - if derivative is (w.r.t) z
            #   - then maybe: [ - 2 (y - z) * d(predicted) (w.r.t) d(b) ]
            #
            # we want d(sse) == 0
            # d (derivative) of predicted (w.r.t) d(b) = d (w.r.t) d(b) * activation function output

            # stepsize = (summed output of [ d(sse) / d(b) ]) * (learning rate eta)
            # new b = old b - stepsize

            # diff, dimensions = n
            # transpose x to line up dimensions (d x n)
            bias += self.eta * diff.sum()
            w += self.eta * x.T.dot(diff)

        return w

    """
        wrapper to calculate and update weights for our model
        
        arguments
        - df: dataframe, contains all columns
        
        returns
        - w: final weights set for model
    """
    def initialize(self):
        df = self.df
        label = self.label

        # get classes and the k-value (# class options)
        class_column = df[[label]]
        df_byclass = df.groupby(by = [label], dropna = False)
        classes = list(df_byclass.groups.keys())
        k = len(classes)

        # x = df without class column
        x = df.copy().drop(columns = label).to_numpy()
        d = x.shape[1]

        # create network layers and weights
        for l in range(self.num_hidden_layers):
            layer = []
            for n in range(self.num_hidden_nodes):
                # weight dimensions = d (feature #) + 1 for bias
                w = np.random.uniform(0, 0.1, d + 1)
                layer.append({'weights': w})
            
            self.network['layers'].append(layer)

        for o in range(self.num_output_nodes):
            node = []
            for i in range(self.num_hidden_nodes):
                # weight dimensions = (number hidden nodes) + 1 for bias
                w = np.random.uniform(0, 0.1, self.num_hidden_nodes + 1)
                node.append({'weights': w})

            self.network['outputs'].append(node)

        return self.network

    """
        calculate class prediction given data and weights
        
        arguments
        - x: data without class column
        - w: weights representing our model
        
        returns
        - class predictions flattened 1-d array
    """
    def predict(self, x, w):
        # check if activation function  >= 0
        #  - return 1
        #  - otherwise return 0
        out = self.activation(x, w)
        return np.where(out > 0, 1, 0).flatten()

    """
        run weights with our test data
        
        arguments
        - df: dataframe (with all columns)
        - w: weights representing our model (for a given target class)
        
        returns
        - returns accuracy of prediction with our given dataframe
    """
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