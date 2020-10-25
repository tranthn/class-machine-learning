#!/usr/bin/env python3
import sys
import time
import numpy as np
import pandas as pd
import data_loader as dl
import logistic as lr
from adaline import Adaline

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

# wrapper helpers to reduce code duplication for processing folds
def logistic_helper(data, label = None, eta = 0.005, iterations = 1):
    if label == None:
        label = class_label

    f = 5 # fold-value

    # tracker variables for performance/timing
    perf = []

    # get starting attrs for building tree
    tune = data['tune']
    attrs = tune.drop(columns = [label]).columns.values

    for i in range(f):
        print('\n========= F O L D #{0} ========='.format(i + 1))
        folds = data['folds'].copy()
        holdout = folds[i]
        folds.pop(i) # remove holdout fold
        training = pd.concat(folds) # concat remaining folds to create training set

        # build the logistic regression model
        w = lr.build(training, class_label, eta, iterations)
        accuracy = lr.test(holdout, w, class_label)

        # track results
        perf.append(accuracy)
        print('accuracy:\t{:.0%}'.format(accuracy))

    print('------------')
    print('\n====== PERFORMANCE SUMMARY ======')
    print_helper_classifier(perf, f)

# wrapper for adaline classification to process folds
def adaline_helper(data, label = None, eta = 0.005, multi = False, iterations = 1):
    if label == None:
        label = class_label

    f = 5 # fold-value

    # tracker variables for performance/timing
    perf = []

    # get starting attrs for building tree
    tune = data['tune']
    attrs = tune.drop(columns = [label]).columns.values

    for i in range(f):
        print('\n========= F O L D #{0} ========='.format(i + 1))
        folds = data['folds'].copy()
        holdout = folds[i]
        folds.pop(i) # remove holdout fold
        training = pd.concat(folds) # concat remaining folds to create training set
        accuracy = 0

        # build adaline model, depending on whether there are multiple classes (k > 2) or not
        if (multi):
            ada = Adaline(label, eta, iterations)
            w_map = ada.build(training)
            accuracy_map = ada.test_multi_class_helper(holdout, w_map)

            # grab the accuracies (values) per class and sum them for total accuracy
            accuracy_sum = np.sum(list(accuracy_map.values()))

            # divide by number of class options (keys) to determine average accuracy
            # for the multi-class scenario
            accuracy = accuracy_sum / (len(accuracy_map.keys()))
        else:
            ada = Adaline(label, eta, iterations)
            w_map = ada.build(training)
            accuracy = ada.test(holdout, w_map['main'])

        # track results
        perf.append(accuracy)
        print('accuracy:\t{:.0%}'.format(accuracy))

    print('------------')
    print('\n====== PERFORMANCE SUMMARY ======')
    print_helper_classifier(perf, f)

################ classification data sets ################
print('\n================== BREAST DATA ================== ')
data = dl.get_breast_data()
# logistic_helper(data, 'class', eta = 0.05, iterations = 10)
# adaline_helper(data, class_label, eta = 0.0005, iterations = 10)

print('\n================== GLASS DATA ================== ')
data = dl.get_glass_data()
# logistic_helper(data, 'class', eta = 0.5, iterations = 10)
# adaline_helper(data, class_label, eta = 0.05, multi = True, iterations = 5)

print('\n================== IRIS DATA ================== ')
data = dl.get_iris_data()
# logistic_helper(data, 'class', eta = 0.01, iterations = 10)
# adaline_helper(data, class_label, eta = 0.005, multi = True, iterations = 10)

print('\n================== SOYBEAN DATA ================== ')
data = dl.get_soy_data()
# logistic_helper(data, 'class', eta = 0.05, iterations = 10)
# adaline_helper(data, class_label, eta = 0.01, multi = True, iterations = 10)

print('\n================== HOUSE VOTING DATA ================== ')
data = dl.get_house_data()
# logistic_helper(data, 'class', eta = 0.3, iterations = 10)
# adaline_helper(data, class_label, eta = 0.01, iterations = 10)