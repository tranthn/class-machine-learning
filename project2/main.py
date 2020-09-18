#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import data_loader as dl

# field name that maps to the class column for data
# all data sets used here will be "class", but is externalized
# to make it easier if code needs to handle different class column names
class_label = 'class'

########################################
## house data
print('\n============== HOUSE DATA ============== ')
data = dl.get_house_data()
# print(data)

########################################
## glass data
print('\n============== GLASS DATA ============== ')
data = dl.get_glass_data()
# print(data)

print('\n============== SEGMENTATION DATA ============== ')
data = dl.get_segmentation_data()
# print(data)

################# regression data sets #################
print('\n============== ABALONE DATA ============== ')
data = dl.get_abalone_data()
test = data['test']

fold = dl.sample_regression_data(test, 'rings', 0)
fold = dl.sample_regression_data(test, 'rings', 1)

print('\n============== FOREST FIRES DATA ============== ')
# data = dl.get_forest_fires_data()
# print(data)

print('\n============== MACHINE DATA ============== ')
# data = dl.get_machine_data()
# print(data)

