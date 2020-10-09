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
    def __init__(self, data):
        # stores feature name to the string query values used for splitting logic
        # meant to keep the query logic we use in entropy for re-use in the tree build
        # and tree prediction later on
        self.feature_map = {}
        self.data = data

    # binary split for numeric columns
    #
    # returns
    #   - median point
    def get_numeric_split(self, df, feature):
        sorted = df.sort_values(by = feature)
        md1 = int(df.shape[0] / 2)
        md2 = int(md1 + 1)

        values = sorted[feature].to_numpy()
        split = (values[md1] + values[md2]) / 2

        med = df[feature].median()

        return med

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

    def mse(self, df, label):
        # best guess based on df passed in (which is split based on feature value)
        avg = df[label].mean()
        rcount = df.shape[0]
        vals = np.full([rcount], avg)
        mse = np.mean((np.subtract(vals, df[label])) ** 2)
        print('mse', mse)

        return mse

    def get_feature_mse(self, df, feature, label):
        split_pt = self.get_numeric_split(df, feature)
        split_point = self.get_numeric_split(df, feature)
        q_lte = self.build_query_string([feature], ['<='], [split_point])
        q_gt = self.build_query_string([feature], ['>'], [split_point])
        feature_opts = [q_lte, q_gt]

        lowest_mse = math.inf
        for f in feature_opts:
            df_filtered = df.query(f)
            mse = self.mse(df_filtered, label)
            if (mse < lowest_mse):
                lowest_mse = mse
        
        return lowest_mse

    # find best feature to be root node
    #
    # returns
    #   - name of best feature
    def pick_best_feature(self, df, label):
        tot = df.shape[0]
        features = df.columns.drop(label)
        best_feature = ""
        lowest_mse = math.inf
        for f in features:
            mse = self.get_feature_mse(df, f, label)
            if (mse < lowest_mse):
                lowest_mse = mse
                best_feature = f

        print()
        print('best feature', best_feature)
        print('lowest mse', lowest_mse)
        return best_feature

    def predict(self, node, row):
        # convert Series to DataFrame to allow query() to use
        # the stored transition string (which looks like `condition` operator value)
        # e.g: `clump-thickness` > 4.0
        if (isinstance(row, pd.Series)):
            row = row.to_frame().T # transpose series to frame

        # if node is leaf, return its decision
        if (node.children is None or len(node.children) == 0):
            return node.decision
        else:
            picked_child = None
            for c in node.children:
                found = row.query(c.transition)
                take_branch = found.shape[0] > 0
                if take_branch:
                    picked_child = c
                    continue
            
            return self.predict(picked_child, row)

    # method to assess accuracy of the tree given some dataframe
    #
    # arguments
    #   - tree: the trained tree
    #   - df: the set we're testing
    #   - label: the name of class column
    def test_tree(self, tree, df, label):
        sself = self
        def check_if_right(tree, row):
            out = sself.predict(tree, row)
            return out == row[label]

        df['right'] = df.apply(lambda row : check_if_right(tree, row), axis = 1)
        print()
        tot = df.shape[0]
        right = df[df['right'] == True].shape[0]
        s = 'tree accuracy: {:.0%}'.format(right / tot)
        print(colored(s, 'green'))

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
    def reg_tree(self, df, label, tree = None, features = None, prior_value = None):
        # default root node
        root_node = Node(items = df, transition = prior_value)

        # if tree hasn't started yet, set to current root
        if (tree == None):
            tree = root_node

        # label decision on node
        if (len(features) <= 1):
            return tree

        # initialize root node with actual feature
        root = self.pick_best_feature(df, label)
        root_node.feature = root
        tree = root_node

        # recalculate groupings to create children nodes
        grouping = df.groupby(by = [root], dropna = False).size()
        feature_opts = grouping.index

        # make leaf node with decision
        if (len(feature_opts) == 1):
            print('only 1 feature option for ', feature_opts)
            return tree

        # there are more than 1 features left
        next_features = np.delete(features, np.where(features == [root]))
        feature_opts = self.feature_map[root]

        for f in feature_opts:
            # query will filter dataframe based on the query condition
            df_filtered = df.query(f)
            df_byclass = df_filtered.groupby(by = [label], dropna = False)
            classes = list(df_byclass.groups.keys())

            # feature-attr pair only represents 1 class, so it'll become leaf
            if (len(classes) == 1):
                # get the only represented class
                dec = classes[0]
                leaf = Node(feature = root, transition = f, decision = dec)
                tree.append_child(leaf)

            # otherwise, feature-attr pair will be recursively split
            else:
                subset = df_filtered
                subset = subset.drop(columns = root)
                subtree = self.reg_tree(subset, label, tree, next_features, f)
                tree.append_child(subtree)

        return tree