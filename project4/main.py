#!/usr/bin/env python3
import sys
import time
import numpy as np
import pandas as pd
import data_loader as dl
import logistic as lr
import adaline as ada

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
# optional execution of condensed or edit methods
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

################ classification data sets ################

print('\n================== BREAST DATA ================== ')
data = dl.get_breast_data()
training = data['folds'][0]
holdout = data['folds'][1]
# logistic_helper(data, 'class', eta = 0.05, iterations = 10)

w = ada.build(training, class_label, eta = 0.05, iterations = 1)
ada.test(holdout, w, class_label)

print('\n================== GLASS DATA ================== ')
data = dl.get_glass_data()
# logistic_helper(data, 'class', eta = 0.5, iterations = 10)

print('\n================== IRIS DATA ================== ')
data = dl.get_iris_data()
# logistic_helper(data, 'class', eta = 0.01, iterations = 10)

print('\n================== SOYBEAN DATA ================== ')
data = dl.get_soy_data()
# logistic_helper(data, 'class', eta = 0.05, iterations = 10)

print('\n================== HOUSE VOTING DATA ================== ')
data = dl.get_house_data()
# logistic_helper(data, 'class', eta = 0.3, iterations = 10)