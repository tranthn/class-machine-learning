#!/usr/bin/env python3
import sys
import math
import numpy as np
import pandas as pd
from termcolor import colored
from node import Node

###### algorithm steps ######
"""
1: iterate over features
    - find best split for a given feature f (midpoint split or between each adjacent TBD)
        - best split is determined by minimal MSE for predictor target class
    - store whichever feature F currently has minimal MSE
2: use that best feature F as root
    - repeat step 1, but TBD if we remove feature F from further consideration
3: keep building tree, we stop splitting if our remaining branch items drop below a preset bin size

early stopping
* stopping criterion
    - threshold for acceptable error (when MSE drops below, stop growing tree)
* lower threshold/error rate results in larger tree (greater height/depth)
    - relaxing error threshold (increasing it) will shrink the tree
"""

class RegressionTree():
    def __init__(self, validation_set = None, threshold = 0, node_min = 20):
        # stores feature name to the string query values used for splitting logic
        # meant to keep the query logic we use during split for re-use in the tree build
        # and tree prediction later on
        self.feature_map = {}

        # used as threshold to stop growing tree once MSE drops enough
        self.threshold = threshold

        # the minimum number of items left in a branch to continue splitting
        # if the number of items < node_min, we stop growing that branch
        self.node_min = node_min

        # validation set used for early stopping as we build the tree
        self.validation_set = validation_set

        # track number of nodes appended
        self.num_nodes = 0

    # binary split for numeric columns
    #
    # returns
    #   - median point
    def get_numeric_split(self, df, feature):
        sorted = df.sort_values(by = feature)
        df_len = df.shape[0]
        split = None
        values = sorted[feature].to_numpy()

        # handle scenarios where we're trying to split on df with only 1 row or 2 rows
        if (df_len == 1):
            split = df[feature].values[0]
        elif (df_len == 2):
            split = (values[0] + values[1]) / 2
        else:
            md1 = int(df_len / 2)
            md2 = md1 + 1
            split = (values[md1] + values[md2]) / 2

        return round(split, 5)

    # check if feature column is categorical
    def is_categorical(self, df, feature):
        ftype = df[feature].dtype
        categorical = ftype != 'float64' and ftype != 'int64'
        return categorical

    # builds query string to use in conjunction with pandas.query()
    # example: `clump-thickness` > 4.0
    def build_query_string(self, features, operators, values):
        query_str = ""
        stop = len(features) - 1

        for idx, f in enumerate(features):
            v = values[idx]

            # must escape categorical / string values, so that
            # pandas.query() does not interpret value as another column name
            if (type(v) is str):
                v = "'{0}'".format(v)

            # {0} - feature name, is escaped within str format, for safety
            q = "`{0}` {1} {2}".format(features[idx], operators[idx], v)
            query_str += q
            if (idx < stop):
                query_str = query_str + " and "

        return query_str

    # helper that calculates MSE for a dataframe
    # it will take in a filtered dataframe and find its average target value
    # the average is compared against each row's actual target value to find MSE
    def calc_mse(self, df, label):
        # avg is based on df passed in (which is pre-split based on feature value)
        avg = df[label].mean()
        rcount = df.shape[0]
        vals = np.full([rcount], avg)
        mse = np.mean((vals - df[label]) ** 2)
        return round(mse, 5)

    # get the minimal MSE for a given feature
    # will use the numeric split function to do
    # a binary split and then pick the lowest MSE of the split groups
    #
    # arguments
    #   - df
    #   - feature: current column
    #   - label: target predictor column, used to assess MSE
    # 
    # returns
    #   - lowest MSE, feature condition that resulted in MSE
    def get_feature_mse(self, df, feature, label):
        # will store query strings we'll use for pandas.query()
        feature_opts = []

        if (self.is_categorical(df, feature)):
            raw_opts = df[feature].unique()
            for o in raw_opts:
                qry = self.build_query_string([feature], ['=='], [o])
                feature_opts.append(qry)

        else:
            split_pt = self.get_numeric_split(df, feature)
            split_point = self.get_numeric_split(df, feature)
            q_lte = self.build_query_string([feature], ['<='], [split_point])
            q_gt = self.build_query_string([feature], ['>'], [split_point])
            feature_opts = [q_lte, q_gt]

        lowest_mse = math.inf
        best_feature_condition = ''

        ## feature_opts is genericized to being query string for filtering
        ## this allows handling when val == [categorical value]
        ## as well as val <= or > [numerical split]
        self.feature_map[feature] = feature_opts

        # evaluate which feature-value option has the optimal/lowest mse
        for f in feature_opts:
            df_filtered = df.query(f)
            mse = self.calc_mse(df_filtered, label)
            if (mse < lowest_mse):
                lowest_mse = mse
                best_feature_condition = f

        # return the lowest MSE as well as the feature condition that yielded this MSE
        # feature_condition will take format for pandas.query, i.e. `sex == 'I'`
        return { 'mse': lowest_mse, 'feature_condition': f }

    # find best feature to be root node
    #
    # returns
    #   - name of best feature
    def pick_best_feature(self, df, label):
        tot = df.shape[0]
        features = df.columns.drop(label)
        best_feature = None
        lowest_mse = math.inf

        for f in features:
            mse_result = self.get_feature_mse(df, f, label)
            mse = mse_result['mse']
            if (mse < lowest_mse):
                lowest_mse = mse
                best_feature = f

        return best_feature

    # traverse a given node to pick best decision for a given data row
    # will recursively traverse down a node
    #
    # arguments:
    #   - tree node: representing a given node at any point in a tree
    #   - row: the data instance we're predicting on
    #
    # returns
    #   - the decision / outcome based on the final leaf node
    def predict(self, node, row):
        # convert Series to DataFrame to allow query() to use
        # the stored transition string (which looks like `condition` operator value)
        # e.g: `clump-thickness` > 4.0
        if (isinstance(row, pd.Series)):
            row = row.to_frame().T # transpose series to frame

        # if node is leaf, return its decision
        if (node.children is None or len(node.children) == 0):
            return node.decision

        # else iterate through the children to see if there
        # is an available branch to take for our row
        else:
            picked_child = None
            for c in node.children:
                found = row.query(c.transition)
                take_branch = found.shape[0] > 0
                if take_branch:
                    picked_child = c
                    continue

            # handle cases where the trained tree does not have a branch
            #   for a particular categorical feature option
            # in this scenario, we'll just end early at current node
            if (picked_child == None):
                return node.decision
            else:            
                return self.predict(picked_child, row)

    # method to assess accuracy of the tree given some dataframe
    #
    # arguments
    #   - tree: the trained tree
    #   - df: the set we're testing
    #   - label: the name of class column
    def test_tree(self, tree, df, label):
        df = df.copy()
        df['guess'] = df.apply(lambda row : self.predict(tree, row), axis = 1)
        mse = np.mean((df['guess'] -  df[label]) ** 2)
        # result = 'tree mse: {0:.5g}'.format(mse)
        # print(colored(result, 'green'))
        return round(mse, 5)

    # method that builds up the regression tree
    # recursively builds down tree as it splits the data
    #
    # arguments
    #   - df
    #   - label
    #   - tree: recursively built, starts at None, but appended to during each call
    #   - features: remaining features to be assessed for splits, reduced on each call
    #   - prior_value: the transition query (condition) to lead to this node
    #           i.e. relative to parent (feature node), what condition lead to this branch call
    #
    # returns
    #   - the tree as its being built up, returns based on conditions above
    def reg_tree(self, df, label, tree = None, prior_value = None):
        # default root node
        root_node = Node(items = df, transition = prior_value)

        # initialize root node with actual feature
        root = self.pick_best_feature(df, label)
        root_node.feature = root

        # set the default decision for any given node to be average of its current item set
        root_node.decision = df[label].mean().round(5)
        tree = root_node

        ## early stopping handling
        if (self.threshold > 0):
            mse = self.test_tree(tree, self.validation_set, label)
            # early stop: if tree's mse drops below threshold, stop growing
            if (mse < self.threshold):
                return tree

        # get the feature's transition options 
        # e.g. `shell_weight` > 0.58 or `shell_weight` <= 0.58
        feature_opts = self.feature_map[root]

        for f in feature_opts:
            # query will filter dataframe based on the query condition
            df_filtered = df.query(f)

            if (df.shape[0] <= self.node_min):
                # if remaining data items are less than or equal 20
                # then we won't split anymore, return tree
                dec = df[label].mean().round(5)
                leaf = Node(feature = root, transition = f, decision = dec)
                tree.append_child(leaf)
                self.num_nodes += 1
            else:
                # otherwise, feature-attr pair will be recursively split
                subset = df_filtered
                subtree = self.reg_tree(subset, label, tree, f)
                tree.append_child(subtree)
                self.num_nodes += 1

        return tree