#!/usr/bin/env python3
import sys
import csv
import numpy as np
import pandas as pd

# m-estimate for pseudo-examples
m_default = 1

# probability to use in our smoothing function
p_default = 0.01

# builds probability table based on class membership and feature values
#
# arguments
#   - row = a given instance of data, whose values we will use to modify the weights
#   - label = name of the column of dataframe that maps to the label/class
#   - m = m-estimate for pseudo-examples
#   - p = probability to use in our smoothing function
#
# returns
#   - probability_df = dataframe representing class and feature probabilities
def build_probability_table(df, label, m = m_default, p = p_default):
    total = df.shape[0]

    ## create data frame to store probabilities
    probability_df = pd.DataFrame()

    cols = df.copy().drop(columns = label).columns.tolist()
    class_opts = df[label].value_counts()
    probability_df = pd.DataFrame(0, index = class_opts.index, columns = cols)
    probability_df.insert(loc = 0, column = 'class%', value = 0)

    for option in class_opts.index:
        df2 = df.copy()

        # grab the rows with that match index == c (class option)
        df2 = df2[df2[label] == option]
        class_total = df2.shape[0]
        probability_df.loc[option,'class%'] = class_total / total

        for feat in cols:
            # probability that feature = 1
            # proability that feature = 0 will just be 1 - prob
            prob = (df2[feat].values == 1).sum()
            probability_df.loc[option, feat] = (prob + m * p) / (class_total + m)

    return probability_df

# helper method
# find the probability of instance belonging to class represented by prob_arr
# 
# arguments
#   - instance = a given instance of data
#   - prob_arr = serialized 1-d array that holds probabilities of features for a particular class
#
# returns
#   - prob = probability that instance belongs to class represented by prob_arr
def compute_probability(instance, prob_arr):
    prob_len = len(prob_arr)
    prob = 1.0
    if (len(instance) == prob_len):
        for idx in range(0, prob_len - 1):
            # prob_arr[i] stores P(f = 1 | c)
            if (instance[idx] == 1):
                prob = prob  * prob_arr[idx]

            # 1 - prob_arr[i] gives us P(f = 0 | c)
            else:
                prob = prob * (1 - prob_arr[idx])
    else:
        print("ARRAY LENGTH MISMATCH")

    return prob

# given data row, will find the membership class with maximal probability
# 
# arguments
#   - row = a given instance of data
#   - probability_df = table of probabilities representing our Naive Bayes model
#
# returns
#   - outcome object = a dict-object representing the class choice/label
#                      and its corresponding probability value
def check_instance(row, probability_df, class_opts):
    choice = None
    prob_max = 0.0
    for c in class_opts.index:
        class_per = probability_df.loc[c, 'class%']
        probs = probability_df.drop(columns = 'class%').to_numpy()[0]
        prob = class_per * compute_probability(row, probs)
        
        if (prob > prob_max):
            prob_max = prob
            choice = c

    return {'choice': c, 'probability': prob_max}

# tests a data frame against a pre-built Naive Bayes probability table
# 
# arguments
#   - df = a given instance of data
#   - probability_df = table of probabilities representing our Naive Bayes model
#   - label = name of the column of dataframe that maps to the label/class
#
# returns
#   - None
def test_model(df, probability_df, label):
    correct = 0
    wrong = 0

    class_opts = df[label].value_counts()
    for _, row in df.iterrows():
        expected = row[label]
        row = row.drop(labels = label)
        outcome = check_instance(row, probability_df, class_opts)
        
        if (outcome['choice'] == expected):
            correct += 1
        else:
            wrong += 1
        
    print('\nBAYES SUMMARY')
    print('-- prediction for class: ', label)
    print('------------------')
    print('Correct\t', correct)
    print('Wrong\t', wrong)