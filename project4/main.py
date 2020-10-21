#!/usr/bin/env python3
import sys
import time
import numpy as np
import pandas as pd
import data_loader as dl
import logistic as lr

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

    print('------\navg. % accuracy:\t\t{:.0%}'.format(avg / fold))

# wrapper helpers to reduce code duplication for processing folds
# optional execution of condensed or edit methods
def classification_helper(data, label = None):
    if label == None:
        label = class_label

    f = 5 # fold-value

    # tracker variables for performance/timing
    perf = []
    pruned_perf = []
    elapsed_time = 0

    # get starting attrs for building tree
    tune = data['tune']
    attrs = tune.drop(columns = [label]).columns.values

    for i in range(f):
        print('\n========= F O L D #{0} ========='.format(i))
        folds = data['folds'].copy()
        holdout = folds[i]
        folds.pop(i) # remove holdout fold
        training = pd.concat(folds) # concat remaining folds to create training set

        # build the model
        # model = xyz(df = training, label = label, tree = None, features = attrs)
        
        # time model testing
        start_time = time.time()
        # result = abc.test_tree(model, holdout, label)
        result = 'fake'
        elapsed = time.time() - start_time
        elapsed_time += elapsed

        # track results
        perf.append(result)

        print('accuracy:\t\t{:.0%}'.format(result))
        print("runtime:\t\t{:.2f}s".format(elapsed))

    print('------------')
    print('\n====== PERFORMANCE SUMMARY ======')
    print_helper_classifier(perf, f)

################ classification data sets ################

print('\n================== BREAST DATA ================== ')
data = dl.get_breast_data()
w = lr.gradient_descent_binary(data['tune'], class_label)
print(w)

print('\n================== GLASS DATA ================== ')
data = dl.get_glass_data()
w = lr.gradient_descent_multi(data['tune'], class_label)
print(w)

print('\n================== IRIS DATA ================== ')
data = dl.get_iris_data()

print('\n================== SOYBEAN DATA ================== ')
data = dl.get_soy_data()

print('\n================== HOUSE VOTING DATA ================== ')
data = dl.get_house_data()