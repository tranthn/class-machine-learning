#!/usr/bin/env python3
import numpy as np
import pandas as pd

global print_delta
print_delta = False

global print_weights
print_weights = False

global print_loss
print_loss = False

global print_output
print_output = False

global print_diff
print_diff = False

# Neural Network that handles both classification and regression scenarios
class NeuralNet():
    def __init__(self, df = None, label = '', eta = 0.05,
                    iterations = 1, layer_structure = [1, 2],
                    regression = False):

        self.df = df
        self.label = label
        self.eta = eta
        self.iterations = iterations
        self.regression = regression

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
        # check if last layer and regression, so we'll use linear
        if (i < (self.num_layers - 1) and self.regression):
            return lambda x: 1
        else:
            return self.sigmoid_derivative

    # run our data instances x through existing network
    def forward_feed(self, x):
        global print_output

        n = x.shape[0]
        layers = self.network['layers']

        for i in range(self.num_layers):
            activation = self.get_activation_fn(i)
            l = layers[i]
            next_input = []

            for j in range(len(l)):
                node = l[j]
                w = node['weights']
 
                # sum of z becomes the input (or x) that we plug into activation function
                z = self.get_dot(x, w)
                output = activation(z)
                node['y'] = output

                # next input column dimensions must align to next layer
                next_input.append(output)

            # reshape raw input to match current layer's # nodes)
            n_nodes = self.layer_structure[i]
            next_input = np.reshape(next_input, (n, n_nodes))
            x = next_input

        return x

    """
        calculate the error from network, in reverse layer order
            - calculates error per node in each layer
            - uses node errors and derivative to set node delta
            - node delta will be used to update weights
    """
    def backpropagate(self):
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

                    # for classification, self.y has dimensions n x k
                    # the # of output nodes = k, process one node / column at a time
                    if not self.regression:
                        y = self.y[:,j]
                    else:
                        y = self.y

                    diff = y - node['y']

                    errors.append(diff)

                    if print_diff:
                        print('y', y)
                        print('node.y', node['y'])
                        print('diff', diff)
                        print()
                print_diff = False

            # not the last layer, so we can look forward to next layer
            else:
                for j in range(len(l)):
                    error = 0
                    next_layer = self.network['layers'][i + 1]

                    for node in next_layer:
                        error += node['weights'][j] * node['delta']

                    errors.append(error)

            for j in range(len(l)):
                node = l[j]

                # get error for this node
                node['delta'] = errors[j] * derivative(node['y'])
                if print_delta and i == self.num_layers - 1:
                    print('node.delta', node['delta'])
                    print()

            print_delta = False

    def calculate_loss(self, y, z):
        global print_loss
        # use MSE for regression and SSE for classification
        if self.regression:
            loss = np.mean((np.subtract(y, z)) ** 2)
        else:
            diff = (y - z)
            loss = (diff ** 2).sum()

        if print_loss:
            print('loss', loss)
        return loss

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
        global print_weights
        layers = self.network['layers']
        eta = self.eta

        # dimensions are [n x d], rotate for consistent alignment
        x = self.x.T

        for i in range(self.num_layers):
            l = layers[i]
            if i > 0:
                prev_layer = layers[i - 1]
                x = [node['y'] for node in prev_layer]
                x = np.array(x)

            for j in range(len(l)):
                node = l[j]

                # update main weights, node['delta'] dimensions are (n, )
                # node['delta'] = errors[j] * self.activation_derivative(node['y'])
                gradient = np.dot(x, node['delta'])
                print('node.delta', node['delta'].shape)
                node['weights'][:-1] += eta * gradient
                
                if print_weights:
                    print('weights', node['weights'][0])
                    print_weights = False

                # update bias values, delta shape is (n, )
                # bias value is singular, so we'll sum delta to adjust
                step = eta * node['delta'].sum()
                node['weights'][-1] += step

    """
        main entry point to build train neural network
            - initializes neural network structure
            - runs forward feed, backprop, and weight updates
    """
    def build(self):
        global print_delta
        global print_weights
        global print_loss
        global print_output
        global print_diff

        # network base structure
        self.network = {
            'input': None,
            'layers': [],
            'outputs': []
        }

        # initialize and fill in network structure
        self.initialize()

        for i in range(self.iterations):
            # print_delta = True
            # print_weights = True
            # print_loss = True
            # print_output = True
            # print_diff = True
            # print_outp = True

            print('ITERATION ', i)
            output = self.forward_feed(self.x)
            loss = self.calculate_loss(self.y, output)
            self.backpropagate()
            self.update_weights()
            print('================\n')

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

    # helper to run classification test set
    def _test_classification(self, df):
        n = df.shape[0]
        classes = df[self.label]
        x = df.copy().drop(columns = self.label).to_numpy()
        y = df[self.label]

        output = self.predict(x)
        predictions = np.argmax(output, axis = 1)

        # convert classes to factors, to use with predictions
        vals, levels = pd.factorize(classes)
        predictions_with_class = levels.take(predictions)
        corr = np.equal(classes.to_numpy(), predictions_with_class).sum()
        print('\nclassification accuracy: {:.2%}'.format(corr / n))
        return (corr / n)

    # helper to run regression test set
    def _test_regression(self, df):
        n = df.shape[0]
        actual = df[self.label]
        x = df.copy().drop(columns = self.label).to_numpy()
        y = df[self.label].to_numpy()

        output = self.predict(x)
        loss = self.calculate_loss(y, output)
        print('\nregression loss: {:.2g}'.format(loss))