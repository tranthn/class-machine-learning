#!/usr/bin/env python3
import random
from collections import Counter # Used for counting
 
# File name: five_fold_stratified_cv.py
# Author: Addison Sears-Collins
# Date created: 7/7/2019
# Python version: 3.7
# Description: Implementation of five-fold stratified cross-validation
# Divide the data set into five random groups. Make sure 
# that the proportion of each class in each group is roughly equal to its 
# proportion in the entire data set.
 
# Required Data Set Format for classification Problems:
# Columns (0 through N)
# 0: class
# 1: Attribute 1 
# 2: Attribute 2
# 3: Attribute 3 
# ...
# N: Attribute N
 
def get_five_folds(instances):
    """
    Parameters:
        instances: A list of dictionaries where each dictionary is an instance. 
            Each dictionary contains attribute:value pairs 
    Returns: 
        fold0, fold1, fold2, fold3, fold4
        Five folds whose class frequency distributions are 
        each representative of the entire original data set (i.e. Five-Fold 
        Stratified Cross Validation)
    """
    # Create five empty folds
    fold0 = []
    fold1 = []
    fold2 = []
    fold3 = []
    fold4 = []
 
    # Shuffle the data randomly
    random.shuffle(instances)
 
    # Generate a list of the unique class values and their counts
    classes = []  # Create an empty list named 'classes'
 
    # For each instance in the list of instances, append the value of the class
    # to the end of the classes list
    for instance in instances:
        classes.append(instance['class'])
 
    # Create a list of the unique classes
    unique_classes = list(Counter(classes).keys())
 
    # For each unique class in the unique class list
    for uniqueclass in unique_classes:
 
        # Initialize the counter to 0
        counter = 0
         
        # Go through each instance of the data set and find instances that
        # are part of this unique class. Distribute them among one
        # of five folds
        for instance in instances:
 
            # If we have a match
            if uniqueclass == instance['class']:
 
                # Allocate instance to fold0
                if counter == 0:
 
                    # Append this instance to the fold
                    fold0.append(instance)
 
                    # Increase the counter by 1
                    counter += 1
 
                # Allocate instance to fold1
                elif counter == 1:
 
                    # Append this instance to the fold
                    fold1.append(instance)
 
                    # Increase the counter by 1
                    counter += 1
 
                # Allocate instance to fold2
                elif counter == 2:
 
                    # Append this instance to the fold
                    fold2.append(instance)
 
                    # Increase the counter by 1
                    counter += 1
 
                # Allocate instance to fold3
                elif counter == 3:
 
                    # Append this instance to the fold
                    fold3.append(instance)
 
                    # Increase the counter by 1
                    counter += 1
 
                # Allocate instance to fold4
                else:
 
                    # Append this instance to the fold
                    fold4.append(instance)
 
                    # Reset the counter to 0
                    counter = 0
 
    # Shuffle the folds
    random.shuffle(fold0)
    random.shuffle(fold1)
    random.shuffle(fold2)
    random.shuffle(fold3)
    random.shuffle(fold4)
 
    # Return the folds
    return  fold0, fold1, fold2, fold3, fold4