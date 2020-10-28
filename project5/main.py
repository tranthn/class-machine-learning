#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import data_loader as dl
import neural_net as nn

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
data = dl.get_breast_data()
print(data['tune'])

print('\n============== GLASS DATA ============== ')
data = dl.get_glass_data()
print(data['tune'])

print('\n============== SOYBEAN DATA ============== ')
data = dl.get_soy_data()
print(data['tune'])

################# regression data sets #################
print('\n============== ABALONE DATA ============== ')
data = dl.get_abalone_data()
print(data['tune'])

print('\n============== MACHINE DATA ============== ')
data = dl.get_machine_data()
print(data['tune'])

print('\n============== FOREST FIRE DATA ============== ')
data = dl.get_forest_fires_data()
print(data['tune'])