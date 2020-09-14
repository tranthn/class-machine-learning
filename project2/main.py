#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import data_loader

# field name that maps to the class column for data
# all data sets used here will be "class", but is externalized
# to make it easier if code needs to handle different class column names
class_label = 'class'

########################################
## house data
print ('\n============== HOUSE DATA ============== ')
data = data_loader.get_house_data()
print(data)

########################################
## glass data
print ('\n============== GLASS DATA ============== ')
data = data_loader.get_glass_data()
print(data)

print ('\n============== ABALONE DATA ============== ')
data = data_loader.get_abalone_data()
print(data)

print ('\n============== FOREST FIRES DATA ============== ')
data = data_loader.get_forest_fires_data()
print(data)

print ('\n============== MACHINE DATA ============== ')
data = data_loader.get_machine_data()
print(data)

print ('\n============== SEGMENTATION DATA ============== ')
data = data_loader.get_segmentation_data()
print(data)