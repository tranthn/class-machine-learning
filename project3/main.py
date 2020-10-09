#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import data_loader as dl
from id3 import ID3Tree
from regression import RegressionTree

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

def printer_helper_regressor(perf, fold):
    avg = 0
    for i, k in enumerate(perf):
        pstr = "{0:.3g}".format(k)
        print('fold #{0}, MSE\t{1}'.format(i, pstr))
        avg += k

    print('------\navg. MSE:\t{0:.3g}'.format(avg / fold))

# wrapper helpers to reduce code duplication for running knn
# optional execution of condensed or edit methods
def classification_helper(data, k, label = None, tuning = False, 
                            enn = False, cnn = False):

    if label == None:
        label = class_label

    f = 5 # fold-value
    tune = data['tune']

    knn_perf = []

    for i in range(f):
        print('\n======== F O L D #{0} ========'.format(i))

        all_folds = data['folds'].copy()
        holdout = all_folds[i]
        folds = all_folds
        folds.pop(i)
        training = pd.concat(folds)

        # allow parameterized run with tuning or testing sets
        if (tuning):
            test = tune
        else:
            test = holdout

        # running classifer knn algorithms, edited and condensed methods are optional
        knn_results = tree.knn_classifier(training, test, label, k)
        knn_perf.append(knn_results)

    print('-------')
    print('\n======== AVG. SUMMARY OF PERFORMANCE [KNN] ========')
    print_helper_classifier(knn_perf, f)

## regression wrapper helper
def regression_helper(data, k, sigma, label = None, tuning = False, 
                        enn = False, cnn = False, threshold = None):

    if label == None:
        label = class_label
    
    f = 5 # fold-value
    tune = data['tune']

    knn_perf = []

    print('\nT H R E S H O L D\t', threshold)
    print('---')
    for i in range(f):
        print('\n======== F O L D #{0} ========'.format(i))
        training = data['training'].copy()
        holdout = dl.sample_regression_data(training, label, i)
        training = training.drop(holdout.index)

        # allow parameterized run with tuning or testing sets
        if (tuning):
            test = tune
        else:
            test = holdout

        knn_results = tree.knn_regressor(training, test, label, k, sigma)
        knn_perf.append(knn_results)
        
    print('----------')
    print('\n======== AVG. SUMMARY OF PERFORMANCE [KNN] ========')
    printer_helper_regressor(knn_perf, f)

################ classification data sets ################
# print('\n============== DUMMY DATA ============== ')
weather = dl.get_weather()
attrs = weather.drop(columns = ['class']).columns.values
# t = ID3Tree(data = weather)
# tr = t.id3_tree(df = weather, label = class_label, tree = None, features = attrs)
# tr.print()

# print('\n============== BREAST DATA ============== ')
data = dl.get_breast_data()
tune = data['tune']
train = data['folds'][0]
test = data['folds'][1]

"""
tree = ID3Tree(data = train)
attrs = tune.drop(columns = [class_label]).columns.values
trained_tree = tree.id3_tree(df = tune, label = class_label, tree = None, features = attrs)
trained_tree.print()
tree.test_tree(trained_tree, test, class_label)
"""

print('\n============== CAR DATA ============== ')
data = dl.get_car_data()
tune = data['tune']

# print('\n============ SEGMENTATION DATA ============ ')
# data = dl.get_segmentation_data()
# tune = data['tune']

################# regression data sets #################
print('\n============== ABALONE DATA ============== ')
data = dl.get_abalone_data()
tune = data['tune']

train = data['folds'][0]
test = data['folds'][1]
print(test)

reg = RegressionTree()
tranined_tree = reg.reg_tree(df = tune, label = 'rings', tree = None, prior_value = None)
tranined_tree.print()
reg.test_tree(tranined_tree, test, 'rings')

# print('\n============== FOREST FIRE DATA ============== ')
# predictor: area
# data = dl.get_forest_fires_data()

# print('\n============== MACHINE DATA ============== ')
# predictor: prp
# data = dl.get_machine_data()