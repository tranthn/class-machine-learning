#!/usr/bin/env python3
import numpy as np
import pandas as pd

# helpers for printing at different execution points in code
# don't want to print excessively, so using globals to limit
global print_forward
print_forward = False
global print_delta
print_delta = False
global print_weights
print_weights = False
global print_gradient
print_gradient = False
global print_loss
print_loss = False
global print_output
print_output = False
global print_diff
print_diff = False

"""
    Neural Net is a multi-layer perceptron for classification and regression. Network layer structure can
    handle an arbitrary number of hidden layers and nodes.

    Activation function for classification is sigmoidal, while the last layer in regression 
        will use a linear activation.

    Network is run in multiple steps:
        - network = NeuralNet(args...) 
        - network.build()
        - network.test(args...)
"""
class NeuralNet():
    def __init__(self, df = None, label = '', eta = 0.05,
                    iterations = 1, layer_structure = [1, 2],
                    regression = False, print_on = False):

        self.df = df
        self.label = label
        self.eta = eta
        self.iterations = iterations
        self.regression = regression
        self.print_on = print_on

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

        network structure:
            - self.network['layers] - array that holds layers
            - each layer contains array of nodes
            - a node has attributes:
                * weights - weights that node applies to previous layer input
                * y - output value of that node
                * delta - used for gradient calculation
        
        returns
            - the built up network structure
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
        class_column = df[label]

        # set y based on regression versus classification
        if not self.regression:
            self.y = pd.get_dummies(class_column, columns = [label]).to_numpy()
        else:
            self.y = class_column.to_numpy()

        # create network layers and weights
        for l in range(self.num_layers):
            layer = []
            if (l > 0):
                # dimension for weights must be based off number of nodes in prior layer
                d = self.layer_structure[l - 1]

            for n in range(self.layer_structure[l]):
                node = { 'weights': None, 'y': None }
                w = np.random.uniform(0, 0.1, d + 1) # + 1 for bias
                node['weights'] = w
                layer.append(node)

            self.network['layers'].append(layer)

        return self.network

    # helper to print network structure in readable format
    def pretty_print(self):
        print('\nNEURAL NETWORK')
        print('# layers (hidden/output)\t', self.num_layers)
        print('# hidden nodes\t\t\t', self.layer_structure[0:-1])
        print('# output nodes\t\t\t', self.layer_structure[-1])
        print('# input (n x d)\t\t\t', self.x.shape)
        print()

        print('LAYERS')
        print('---------')
        for i in range(self.num_layers):
            print('layer', i)
            print('-----')
            layer = self.network['layers'][i]
            for j in range(len(layer)):
                node = layer[j]
                print('node', j)
                print('weights', node['weights'])
                print('output', node['y'])
                print()
            print('----')

####################################################################################

    """
        compute dot product for dataframe and weights
        
        arguments:
            - x: dataframe without class column, dimensions = n x d
            - w: weights, dimensions = d x 1
        
        returns
            - z: dot product of x * w plus bias value
    """
    def get_dot(self, x, wts):
        # z = output, dimensions = n x 1 or just (n, )
        w = wts[:-1]
        bias = wts[-1]
        z = np.dot(x, w) + bias
        return z

    """
        activation function

        arguments:
            - input: the instance values modified with weights

        returns
            - output of activation function
    """
    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    """
        derivative of activation function, for use with gradient descent

        arguments:
            - input: the instance values modified with weights

        returns
            - output of derivactive function
    """
    def sigmoid_derivative(self, input):
        o = self.sigmoid(input)
        dS = o * (1 - o)
        return dS

    # identity function for linear output
    def identity(self, input):
        return input

    # helper to determine which activation function to use
    def get_activation_fn(self, i):
        # check if last layer and regression, so we'll use linear
        if (i == (self.num_layers - 1) and self.regression):
            return self.identity
        else:
            return self.sigmoid

    # helper to determine which derivative function to use
    def get_activation_derivative_fn(self, i):
        # check if last layer and regression, so we'll return constant for derivative
        if (i == (self.num_layers - 1) and self.regression):
            return lambda x: 1
        else:
            return self.sigmoid_derivative

    """
        feeds an input example x through the network
        
        the input into each subsequent layer will be output of prior layer

        arguments:
            - x: a given data example / row to run through network

        returns:
            - x: the last x value, i.e. the output values of the last layer
    """
    def forward_feed(self, x):
        global print_forward

        n = x.shape[0]
        layers = self.network['layers']

        for i in range(self.num_layers):
            activation = self.get_activation_fn(i)
            l = layers[i]
            next_input = []

            # iterate over nodes within layer
            for j in range(len(l)):
                node = l[j]
                w = node['weights']
 
                # sum of z becomes the input (or x) that we plug into activation function
                z = self.get_dot(x, w)
                output = activation(z)
                node['y'] = output

                if print_forward:
                    print('----->')
                    print('forward feed')
                    print('\ti', i)
                    print('\tj', j)
                    print('\tx', x)
                    print()
                    print('\tw', w)
                    print()
                    print('\tz', z)
                    print('\toutput', output)
                    print('----->')

                next_input.append(output)

            x = next_input

        print_forward = False
        return x

    """
        calculate the error from network, in reverse layer order
            - calculates error per node in each layer
            - uses node errors and derivative to set node delta
            - node delta will be used to update weights

        arguments:
            - index: row index used for classification only, to grab the class vector

        returns: None
    """
    def backpropagate(self, index):
        global print_delta
        global print_diff

        layers = self.network['layers']
        for i in reversed(range(self.num_layers)):
            derivative = self.get_activation_derivative_fn(i)
            l = layers[i]
            errors = []

            # this is the last layer, aka output layer
            if (i == (self.num_layers - 1)):
                for j in range(len(l)):
                    node = l[j]

                    # for classification, grab the one-hot encoded
                    # class vector for the given row by index
                    if not self.regression:
                        y = self.y[index,:]
                    else:
                        y = self.y

                    diff = y[j] - node['y']

                    errors.append(diff)

                    if print_diff:
                        print('\n<----- backprop')
                        print('layer {0}'.format(i))
                        print('\ty', y[j])
                        print('\tnode.y', node['y'])
                        print('\tdiff', diff)
                        print('<-----')

            # not the last layer, so we can look forward to next layer
            else:
                # iterate through inner layer nodes
                for j in range(len(l)):
                    error = 0
                    next_layer = self.network['layers'][i + 1]

                    for node in next_layer:
                        error += node['weights'][j] * node['delta']

                    errors.append(error)

            # set the delta values for a given node
            # since we start with last layer, we know we'll set the delta values backwards too
            for j in range(len(l)):
                node = l[j]
                node['delta'] = errors[j] * derivative(node['y'])

                if print_delta:
                    print('layer {0}, node {1}, delta value: {2}'.format(i, j, node['delta']))

            if print_delta:
                print()
    
        print_delta = False
        print_diff = False

    """
        calculates the loss between expected and neural network output
            - calculates squared sum error (SSE) for classification
            - calculates mean squared error (MSE) for regression

        arguments:
            - y: the expected output values
            - z: the calculated (predicted) output values
        
        returns:
            - loss value
    """
    def calculate_loss(self, y, z):
        global print_loss
        # use MSE for regression and SSE for classification
        if self.regression:
            diff = np.subtract(y, z)
            loss = np.mean((diff) ** 2)
        else:
            diff = (y - z)
            loss = (diff ** 2).sum()

        return loss

    """
        the main function for calculating gradient and updating node weights
            - also updates the bias value that is part of the weight array
        
        arguments
            - x: dataframe without class column
        
        returns: None
    """
    def update_weights(self, x):
        global print_gradient
        global print_weights
        layers = self.network['layers']
        eta = self.eta

        for i in range(self.num_layers):
            l = layers[i]
            if i > 0:
                prev_layer = layers[i - 1]
                x = [node['y'] for node in prev_layer]
                x = np.array(x)

            for j in range(len(l)):
                node = l[j]

                if print_weights:
                    print('\nweights before update:', node['weights'])

                gradients = []
                for f in range(len(x)):
                    

                    # gradient for regression is modified to prevent exploding gradient values
                    if self.regression:
                        gradient = (1 / self.df.shape[0]) * x[f] * node['delta']
                    else:
                        gradient = x[f] * node['delta']

                    gradients.append(gradient)
                    node['weights'][f] += eta * gradient

                # update bias values, delta shape is (n, )
                # bias value is singular, so we'll sum delta to adjust
                step = eta * node['delta']
                node['weights'][-1] += step

                if print_weights and print_gradient:
                    print('\ngradients', gradients)
                    print('\nweights after update:', node['weights'])
                    print_weights = False
                    print_gradient = False

    """
        main entry point to build train neural network
            - initializes neural network structure
            - runs forward feed, backprop, and weight updates for every row in input data
    """
    def build(self):
        global print_forward
        global print_delta
        global print_weights
        global print_loss
        global print_output
        global print_diff
        global print_gradient
        
        if self.print_on:
            print_forward = True
            print_delta = True
            print_weights = True
            print_loss = True
            print_output = True
            print_diff = True
            print_gradient = True

        # network base structure
        self.network = {
            'input': None,
            'layers': [],
            'outputs': []
        }

        # initialize and fill in network structure
        self.initialize()

        for i in range(self.iterations):
            for i, x in enumerate(self.x):
                output = self.forward_feed(x)
                self.backpropagate(i)
                self.update_weights(x)

    """
        runs out input row through the network
        
        arguments
            - x: a given data example / row

        returns
            - prediction for the example, which may be a singular value for regression
                or an array of probabilities for classification
    """
    def predict(self, x):
        out = self.forward_feed(x)
        return out

    """
        run weights with our test data
        
        arguments
            - df: dataframe (with all columns)
            - w: weights representing our model (for a given target class)
        
        returns
            - returns accuracy of prediction with our given dataframe
    """
    def test(self, df):
        if (self.regression):
            results = self._test_regression(df)
        else:
            results = self._test_classification(df)
        
        return results

    """
        helper to run classification test set
            - will take argmax of probabilities output by predict()

        arguments:
            - df: test dataframe
        
        returns:
            - the classification accuracy as percentage value (0.0 - 1.0)
    """
    def _test_classification(self, df):
        n = df.shape[0]
        classes = df[self.label]
        df_x = df.copy().drop(columns = self.label)

        predictions = []
        for _, x in df_x.iterrows():
            output = self.predict(x)
            idx = np.argmax(output)
            predictions.append(idx)

        # convert classes to factors, to use with predictions
        vals, levels = pd.factorize(classes)
        predictions_with_class = levels.take(predictions)
        
        if self.print_on:
            print()
            print('actual\t\t', classes.to_numpy())
            print('predictions\t', predictions_with_class.to_numpy())
            print()

        corr = np.equal(classes.to_numpy(), predictions_with_class).sum()
        return (corr / n)

    """
        helper to run regression test set
        
        arguments:
            - df: test dataframe
        
        returns:
            - the MSE calculated between the actual and predicted target values
    """
    def _test_regression(self, df):
        n = df.shape[0]
        actual = df[self.label]
        df_x = df.copy().drop(columns = self.label)
        y = df[self.label].to_numpy()

        predictions = []
        for _, x in df_x.iterrows():
            output = self.predict(x)
            predictions.append(output) 

        if self.print_on:
            print('test regression')
            print('expected', y)
            print('predicted', predictions)

        loss = self.calculate_loss(y, predictions)
        return loss