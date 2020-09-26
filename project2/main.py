#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import data_loader as dl
import knn

# field name that maps to the class column for data
# all data sets used here will be "class", but is externalized
# to make it easier if code needs to handle different class column names
class_label = 'class'
"""
########################################
## house data
print('\n============== HOUSE DATA ============== ')
data = dl.get_house_data()
folds = data['folds']
tune = data['tune']
test = data['folds'][0]
train = data['folds'][1:]

print('\n---- tuning data ----')
trains = pd.concat(train)

knn_model = knn.knn_classifier(trains, tune, class_label, k = 5)
cnn = knn.condensed_knn(trains, tune, class_label)
enn = knn.edited_knn(trains, tune, class_label)

cnn_model = knn.knn_classifier(cnn, tune, class_label, k = 5)
enn_model = knn.knn_classifier(enn, tune, class_label, k = 5)
# knn_model = knn.knn_classifier(trains, test, class_label, k = 9)

########################################
## glass data
print('\n============== GLASS DATA ============== ')
data = dl.get_glass_data()
folds = data['folds']
tune = data['tune']
test = data['folds'][0]
train = data['folds'][1:]

print('\n---- tuning data ----')
trains = pd.concat(train)

knn_model = knn.knn_classifier(trains, tune, class_label, k = 9)
cnn = knn.condensed_knn(trains, tune, class_label)
enn = knn.edited_knn(trains, tune, class_label)

cnn_model = knn.knn_classifier(cnn, tune, class_label, k = 7)
enn_model = knn.knn_classifier(enn, tune, class_label, k = 7)

# print('\n---- testing data ----')
# knn_model = knn.knn_classifier(trains, test, class_label, k = 9)

print('\n============== SEGMENTATION DATA ============== ')
data = dl.get_segmentation_data()
class_label = 'CLASS'

folds = data['folds']
tune = data['tune']
test = data['folds'][0]
train = data['folds'][1:]

print('\n---- tuning data ----')
trains = pd.concat(train)

knn_model = knn.knn_classifier(trains, tune, class_label, k = 11)
cnn = knn.condensed_knn(trains, tune, class_label)
enn = knn.edited_knn(trains, tune, class_label)

cnn_model = knn.knn_classifier(cnn, tune, class_label, k = 11)
enn_model = knn.knn_classifier(enn, tune, class_label, k = 11)

# print('\n---- testing data ----')
# knn_model = knn.knn_classifier(trains, test, class_label, k = 9)
"""

################# regression data sets #################
print('\n============== ABALONE DATA ============== ')
data = dl.get_abalone_data()
test = data['test']

fold = dl.sample_regression_data(test, 'rings', 0)
fold = dl.sample_regression_data(test, 'rings', 1)
# knn.knn_regressor(fold, test, label = 'rings', k = 10, sigma = 1)

print('\n============== FOREST FIRES DATA ============== ')
# data = dl.get_forest_fires_data()
# print(data)

print('\n============== MACHINE DATA ============== ')
data = dl.get_machine_data()
test = data['test']
# print(test.dtypes)
# print(data)

fold = dl.sample_regression_data(test, 'prp', 1)
# knn.knn_regressor(fold, test, label = 'prp', k = 10, sigma = 0.5)