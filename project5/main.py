#!/usr/bin/env python3
import sys
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
        p = k['incorrect'] / k['total']
        pstr = "{:.0%}".format(p)
        print('fold #{0}, misclassified\t{1}'.format(i, pstr))
        avg += p

    print('------\navg. % misclassified:\t{:.0%}'.format(avg / fold))

def print_helper_regressor(perf, fold):
    avg = 0
    for i, k in enumerate(perf):
        pstr = "{0:.3g}".format(k)
        print('fold #{0}, MSE\t{1}'.format(i, pstr))
        avg += k

    print('------\navg. MSE:\t{0:.3g}'.format(avg / fold))

# wrapper helpers to reduce code duplication for running knn
# optional execution of condensed or edit methods
def classification_helper(data, label = None):
    if label == None:
        label = class_label

    f = 5 # fold-value
    tune = data['tune']
    perf = []

    for i in range(f):
        print('\n======== F O L D #{0} ========'.format(i))

        all_folds = data['folds'].copy()
        holdout = all_folds[i]
        folds = all_folds
        folds.pop(i)
        training = pd.concat(folds)

        # running classifer knn algorithms, edited and condensed methods are optional
        # model = blah
        result = ''
        perf.append(result)

    print('-------')
    print('\n======== AVG. SUMMARY OF PERFORMANCE ========')
    print_helper_classifier(perf, f)

## regression wrapper helper
def regression_helper(data, label = None):
    if label == None:
        label = class_label

    f = 5 # fold-value
    tune = data['tune']
    perf = []

    for i in range(f):
        print('\n======== F O L D #{0} ========'.format(i))

        all_folds = data['folds'].copy()
        holdout = all_folds[i]
        folds = all_folds
        folds.pop(i)
        training = pd.concat(folds)

        # running classifer knn algorithms, edited and condensed methods are optional
        # model = blah
        result = ''
        perf.append(result)

    print('-------')
    print('\n======== AVG. SUMMARY OF PERFORMANCE ========')
    print_helper_regressor(perf, f)

########################################
print('\n============== BREAST DATA ============== ')
# d = 9, k = 2
data = dl.get_breast_data()
df = data['tune']
nn = NeuralNet(df = df, label = 'class', eta = 0.01, iterations = 500, layer_structure = [6, 2])
nn.build()
nn.test(data['folds'][0])

print('\n============== GLASS DATA ============== ')
# d = 9, k = 6
data = dl.get_glass_data()
df = data['tune']
# nn = NeuralNet(df = df, label = 'class', eta = 0.01, iterations = 1000, layer_structure = [8, 6])
# nn.build()
# nn.test(data['folds'][0])

print('\n============== SOYBEAN DATA ============== ')
# d = 73 (includes dummied columns), k = 4
data = dl.get_soy_data()
df = data['tune']
# nn = NeuralNet(df = df, label = 'class', eta = 0.01, iterations = 1000, layer_structure = [50, 4])
# nn.build()
# nn.test(data['folds'][0])

################# regression data sets #################
print('\n============== ABALONE DATA ============== ')
# data = dl.get_abalone_data()
# df = data['tune']

print('\n============== MACHINE DATA ============== ')
# d = 37 (includes dummied columns)
data = dl.get_machine_data()
df = data['tune']
nn = NeuralNet(df = df, label = 'prp', eta = 0.01, iterations = 50, layer_structure = [6, 1], regression = True)
nn.build()
nn.test(data['folds'][0])

print('\n============== FOREST FIRE DATA ============== ')
# data = dl.get_forest_fires_data()
# df = data['tune']