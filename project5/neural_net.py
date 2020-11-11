#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import math

##################################################
bias = 0.01

class NeuralNet():
    def __init__(self, df = None, label = '', eta = 0.05, iterations = 1,
                    layer_structure = [1, 2]):

        self.df = df
        self.label = label
        self.eta = eta
        self.iterations = iterations

        # boundary sizes for our network structure
        # layer structure is array of ints
        #   - length of array = # layers, including output layer
        #   - value at given index is the # nodes in the layer
        self.layer_structure = layer_structure
        self.num_layers = len(layer_structure)

    """
        set up the data structure to represent the neural network
            - creates each layer, each node for a given layer
            - creates the weight array (d + 1) for a node
        
        arguments: none
        
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

        # set x (data frame without label) and y (class labels)
        # classes, one-hot encoded here so that main dataframe class is left alone
        self.x = x
        class_column = df[self.label]
        self.y = pd.get_dummies(class_column, columns = [label]).to_numpy()

        # create network layers and weights
        for l in range(self.num_layers):
            layer = []
            if (l > 0):
                # dimension for weights must be based off number of nodes in prior layer
                d = self.layer_structure[l - 1]

            print('layer l', l)
            print('d', d)
            for n in range(self.layer_structure[l]):
                node = { 'weights': None, 'y': None }

                # weight dimensions = d (feature #) + 1 for bias
                w = np.random.uniform(0, 0.1, d + 1)
                node['weights'] = w
                layer.append(node)

            self.network['layers'].append(layer)

        return self.network

    """
        helper to print network structure in readable format
    """
    def pretty_print(self):
        print('NEURAL NETWORK')
        print('# layers (hidden/output)\t', self.num_layers)
        print('# hidden nodes\t\t\t', self.layer_structure[0:-1])
        print('# output nodes\t\t\t', self.layer_structure[-1])
        print()

        print('LAYERS')
        print('---------')
        for layer in self.network['layers']:
            print('layer')
            for node in layer:
                print('weights', node['weights'].shape)
                print(node['weights'])
                print()
                print('output', node['y'].shape)
                print(node['y'])
                print()
            print('----')

        print('\nOUTPUTS')
        print('---------')
        for o in self.network['outputs']:
            print(o)

####################################################################################

    """
        compute dot product for dataframe and weights
        
        arguments:
        - x: dataframe without class column, dimensions = n x d
        - w: weights, dimensions = d x 1
        
        returns
        - z: dot product of x * w plus bias value
    """
    def activation(self, x, wts):
        # z = output, dimensions = n x 1 or just (n, )
        w = wts[:-1]
        bias = wts[-1]
        z = np.dot(x, w) + bias
        return z

    """
        sigmoid function

        arguments:
            - input: the instance values modified with weights

        returns
            - output of sigmoid function
    """
    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    """
        derivative of sigmoid function, for use with gradient descent

        arguments:
            - input: the instance values modified with weights

        returns
            - output of derivactive function
    """
    def sigmoid_derivative(self, input):
        o = self.sigmoid(input)
        dS = o * (1 - o)
        return dS

    # run our data instances x through existing network
    def forward_feed(self, x):
        n = x.shape[0]
        layers = self.network['layers']

        for i in range(self.num_layers):
            l = layers[i]
            next_input = []
            for j in range(len(l)):
                node = l[j]

                # w dimensions: d + 1
                w = node['weights']
                
                # sum of z becomes the input (or x) that we plug into sigmoid function
                z = self.activation(x, w)
                output = self.sigmoid(z)
                node['y'] = output

                # next input column dimensions must align to next layer
                next_input.append(output)

            # if we have 2 layers, indices = 0, 1
            # 0 is between input and hidden layer 1
            # 1 is between hidden layer 1 and output
            # reshape raw input to match current layer's # nodes
            n_nodes = self.layer_structure[i]
            next_input = np.reshape(next_input, (n, n_nodes))
            x = next_input

        return x

    def backpropagate(self):
        layers = self.network['layers']
        errors = []
        for i in reversed(range(self.num_layers)):
            l = layers[i]

            # if we're on last layer, then we will use the original
            # class values for y, otherwise we use output from nodes
            if (i == (self.num_layers - 1)):
                for j in range(len(l)):
                    node = l[j]

                    # since self.y has dimensions n x k, the # of output nodes = k
                    # so if we're looking at one output node at a time
                    # we need to grab 1 column of y at a time
                    y = self.y[:,j]
                    diff = (y - node['y'])
                    errors.append(diff)

            # this is not the last layer, so we can look forward
            # to next layer to calculate the error
            else:
                for j in range(len(l)):
                    error = 0
                    next_layer = self.network['layers'][i + 1]

                    # look to next layer
                    for node in next_layer:
                        error += node['weights'][j] * node['delta']

                    errors.append(error)

            for j in range(len(l)):
                node = l[j]

                # get error for this node
                node['delta'] = errors[j] * self.sigmoid_derivative(node['y'])

    def calculate_loss(self, y, z):
        # sum of squared errors (or residuals)
        diff = (y - z)
        sse = (diff ** 2).sum()

        return sse

    """
        the main function for calculating gradient and loss
        
        arguments
        - x: dataframe without class column
        - y: class column from dataframe
        - w: weights, may be random weights or the updated weights passed by caller
        
        returns:
        - w: final weights after all iterations
    """
    def update_weights(self):
        layers = self.network['layers']
        eta = self.eta
        
        # dimensions are [n x k], rotate for consistent alignment
        x = self.x.T

        for i in range(self.num_layers):
            l = layers[i]
            if i > 0:
                prev_layer = layers[i - 1]
                x = [node['y'] for node in prev_layer]
                # x's dimensions are [k x n]
                x = np.array(x)

            for j in range(len(l)):
                node = l[j]

                # update main weights
                # node['delta'] dimensions are (n, )
                gradient = np.dot(x, node['delta'])
                node['weights'][:-1] += eta * gradient

                # update bias values
                # delta shape is (n, )
                # bias value is singular, so we'll sum delta to adjust
                step = eta * node['delta'].sum()
                node['weights'][-1] += step

    def build(self):
        # network base structure
        self.network = {
            'input': None,
            'layers': [],
            'outputs': []
        }

        # initialize and fill in network structure
        self.initialize()

        for i in range(self.iterations):
            # run network forward to generate outputs
            output = self.forward_feed(self.x)

            # calculate SSE with this network
            sse = self.calculate_loss(self.y, output)

            # backpropagate
            self.backpropagate()

            # update weights
            self.update_weights()

####################################################################################

    """
        calculate class prediction given data and weights
        
        arguments
        - x: data without class column
        - w: weights representing our model
        
        returns
        - class predictions flattened 1-d array
    """
    def predict(self, x):
        out = self.forward_feed(x)
        predictions = np.argmax(out, axis = 1)
        return predictions

    """
        run weights with our test data
        
        arguments
        - df: dataframe (with all columns)
        - w: weights representing our model (for a given target class)
        
        returns
        - returns accuracy of prediction with our given dataframe
    """
    def test(self, df):
        #### calculate model with samples and weights ####
        # x = df without class column
        n = df.shape[0]
        classes = df[self.label]
        x = df.copy().drop(columns = self.label).to_numpy()
        y = df[self.label]

        ### accuracy map
        accuracy_map = {}

        # predict classes with given weight array
        predictions = self.predict(x)

        # convert classes to factors, to use with predictions
        vals, levels = pd.factorize(classes)
        predictions_with_class = levels.take(predictions)
        comp = np.equal(classes.to_numpy(), predictions_with_class.to_numpy())
        corr = sum(comp)
        print('accuracy: {:.2%}'.format(corr / n))
        return (corr / n)