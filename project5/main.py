#!/usr/bin/env python3
import sys
import time
import numpy as np
import pandas as pd
import data_loader as dl
from neural_net import NeuralNet

# field name that maps to the class column for data
# all data sets used here will be "class", but is externalized
# to make it easier if code needs to handle different class column names
class_label = 'class'

# pretty print the performance objects from cross validation
def print_helper_classifier(perf, fold):
    avg = 0
    for i, k in enumerate(perf):
        pstr = "{:.0%}".format(k)
        print('fold #{0}, accuracy\t{1}'.format(i, pstr))
        avg += k

    print('------\navg. % accuracy:\t{:.0%}'.format(avg / fold))

def print_helper_regressor(perf, fold):
    avg = 0
    for i, k in enumerate(perf):
        pstr = "{0:.5g}".format(k)
        print('fold #{0}, MSE\t{1}'.format(i, pstr))
        avg += k

    print('------\navg. MSE:\t{0:.5g}'.format(avg / fold))

# wrapper helper to run neural network training and test with all folds
def neural_net_helper(data, label = None, eta = 0.01, iterations = 100, layer_structure = [], regression = False):
    if label == None:
        label = class_label

    f = 5 # fold-value
    tune = data['tune']
    perf = []

    start_time = time.time()
    for i in range(f):
        # print('\n======== F O L D #{0} ========'.format(i))

        all_folds = data['folds'].copy()
        holdout = all_folds[i]
        folds = all_folds
        folds.pop(i)
        training = pd.concat(folds)

        nn = NeuralNet(training, label, eta, iterations, layer_structure, regression)
        nn.build()
        result = nn.test(holdout)
        perf.append(result)

    elapsed = time.time() - start_time
    print('\n======== avg. summary of performance ========')

    if regression:
        print_helper_regressor(perf, f)
    else:
        print_helper_classifier(perf, f)

    print('total execution time:\t{:.2f}s'.format(elapsed))

########################################
print('\n============== BREAST DATA ============== ')
# d = 9, k = 2
data = dl.get_breast_data()
print('----------------- 0-layer -----------------')
neural_net_helper(data = data, label = 'class', eta = 0.01, iterations = 100, layer_structure = [2])
print('----------------- 1-layer -----------------')
neural_net_helper(data = data, label = 'class', eta = 0.01, iterations = 100, layer_structure = [6, 2])
print('----------------- 2-layer -----------------')
neural_net_helper(data = data, label = 'class', eta = 0.01, iterations = 100, layer_structure = [6, 4, 2])

print('\n============== GLASS DATA ============== ')
# d = 9, k = 6
data = dl.get_glass_data()
print('----------------- 0-layer -----------------')
neural_net_helper(data = data, label = 'class', eta = 0.01, iterations = 100, layer_structure = [6])
print('----------------- 1-layer -----------------')
neural_net_helper(data = data, label = 'class', eta = 0.01, iterations = 100, layer_structure = [8, 6])
print('----------------- 2-layer -----------------')
neural_net_helper(data = data, label = 'class', eta = 0.01, iterations = 100, layer_structure = [4, 4, 6])

print('\n============== SOYBEAN DATA ============== ')
# d = 73 (includes dummied columns), k = 4
data = dl.get_soy_data()
print('----------------- 0-layer -----------------')
neural_net_helper(data = data, label = 'class', eta = 0.01, iterations = 100, layer_structure = [4])
print('----------------- 1-layer -----------------')
neural_net_helper(data = data, label = 'class', eta = 0.01, iterations = 100, layer_structure = [40, 4])
print('----------------- 2-layer -----------------')
neural_net_helper(data = data, label = 'class', eta = 0.01, iterations = 100, layer_structure = [20, 10, 4])

################# regression data sets #################
print('\n============== ABALONE DATA ============== ')
# d = 11 (2 dummies), regression predictor: rings
data = dl.get_abalone_data()
# print('----------------- 0-layer -----------------')
# neural_net_helper(data = data, label = 'rings', eta = 0.01, iterations = 10, layer_structure = [1], regression = True)
# print('----------------- 1-layer -----------------')
# neural_net_helper(data = data, label = 'rings', eta = 0.01, iterations = 10, layer_structure = [8, 1], regression = True)
# print('----------------- 2-layer -----------------')
# neural_net_helper(data = data, label = 'rings', eta = 0.01, iterations = 10, layer_structure = [8, 6, 1], regression = True)

print('\n============== MACHINE DATA ============== ')
# d = 37 (includes dummied columns), regression predictor: prp
data = dl.get_machine_data()
print('----------------- 0-layer -----------------')
# print(data['folds'])
# neural_net_helper(data = data, label = 'prp', eta = 0.01, iterations = 10, layer_structure = [1], regression = True)
# print('----------------- 1-layer -----------------')
# neural_net_helper(data = data, label = 'prp', eta = 0.01, iterations = 10, layer_structure = [20, 1], regression = True)
# print('----------------- 2-layer -----------------')
# neural_net_helper(data = data, label = 'prp', eta = 0.01, iterations = 10, layer_structure = [20, 6, 1], regression = True)

print('\n============== FOREST FIRE DATA ============== ')
# d = 30 (includes dummied columns), regression predictor: area [of fire]
data = dl.get_forest_fires_data()
# print('----------------- 0-layer -----------------')
# neural_net_helper(data = data, label = 'area', eta = 0.01, iterations = 10, layer_structure = [1], regression = True)
# print('----------------- 1-layer -----------------')
# neural_net_helper(data = data, label = 'area', eta = 0.01, iterations = 10, layer_structure = [20, 1], regression = True)
# print('----------------- 2-layer -----------------')
# neural_net_helper(data = data, label = 'area', eta = 0.01, iterations = 10, layer_structure = [20, 10, 1], regression = True)