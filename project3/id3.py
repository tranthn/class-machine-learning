#!/usr/bin/env python3
import sys
import math
import numpy as np
import pandas as pd
from termcolor import colored
from node import Node

class ID3Tree():
    def __init__(self, validation_set = None):
        # stores feature name to the string query values used for splitting logic
        # meant to keep the query logic we use in entropy for re-use in the tree build
        # and tree prediction later on
        self.feature_map = {}

        # validation set used for early stopping as we build the tree
        self.validation_set = validation_set

        # track number of nodes appended
        self.num_nodes = 0

        # track number of pruned nodes
        self.num_pruned_nodes = 0

    # helper to find most common class in filtered dataframe
    def most_common_class(self, df, label):
        # groups by class and the count of rows that belong in a given class option
        # displays in descending count order, so the first index value corresponds
        # the most common class of this dataframe
        df_grouped = df.groupby(by = [label]).size()
        return df_grouped.index.values[0]

    # helper to check if feature column is categorical
    def is_categorical(self, df, feature):
        ftype = df[feature].dtype
        categorical = ftype != 'float64' and ftype != 'int64'
        return categorical

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
            # otherwise, use the average between the 2 midpoints of current feature colummn
            md1 = int(df_len / 2)
            md2 = md1 + 1
            split = (values[md1] + values[md2]) / 2

        return round(split, 5)

    # calculate individual information gain value for given position / total combo
    #
    # arguments
    #   - total: total # items in a given feature-attr combo
    #   - pos: # of class positive examples with given feature-attr combo
    #
    # returns
    #   - calculated log2 information gain
    def info_gain(self, total, pos):
        # negative count is just remainder
        neg = total - pos
        p_pn = pos / total
        n_pn = neg / total

        if (p_pn == 0 or n_pn == 0):
            return 0
        else:
            i = (-p_pn * math.log(p_pn, 2)) - (n_pn * math.log(n_pn, 2))
            return i

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

    # wrapper that calculates entropy for a given feature
    # holds logic to handle both categorical and numerical columns
    #
    # returns
    #   - the entropy for a given feature
    def entropy(self, df, feature, label, class_opt):
        tot = df.shape[0]
        ent = 0

        feature_opts = []
        if (self.is_categorical(df, feature)):
            raw_opts = df[feature].unique()
            for o in raw_opts:
                qry = self.build_query_string([feature], ['=='], [o])
                feature_opts.append(qry)

        ## numeric: do binary split
        ## create 2 queries, one less than or equal to split pt
        ## and remaining split is greater than split pt
        else:
            split_point = self.get_numeric_split(df, feature)
            q_lte = self.build_query_string([feature], ['<='], [split_point])
            q_gt = self.build_query_string([feature], ['>'], [split_point])
            feature_opts = [q_lte, q_gt]

        ## feature_opts is genericized to being query string for filtering
        ## this allows handling when val == [categorical value]
        ## as well as val <= or > [numerical split]
        self.feature_map[feature] = feature_opts
        for f in feature_opts:
            # query will filter dataframe based on the query condition
            df_filtered = df.query(f)
            pn = df_filtered.shape[0]

            # determine how many classes represented after filtering above
            class_count = len(df_filtered[label].unique())

            # check information gain for a specific class option
            df_filtered_cl = df_filtered[df_filtered[label] == class_opt]
            p = df_filtered_cl.shape[0]

            # check if the feature has a split for predicting class or not
            # if the length = 0, that means this feature-opt combo all fell
            # within the same outcome class, take else branch
            if (class_count > 1):
                if_gain = self.info_gain(pn, p)
                ent += (pn / tot) * if_gain
            else:
                # gain should be 0
                if_gain = 0

        return ent

    # find best feature to be root node
    # uses the entropy helper to determine which feature has best gain
    #
    # returns
    #   - name of best feature
    def pick_best_feature(self, df, label):
        tot = df.shape[0]
        features = df.columns.drop(label)
        class_opts = df[label].unique()

        # calculate set entropy
        grouping = df.groupby(by = [label]).size()
        set_entropy = self.info_gain(tot, grouping[1])
        best_feat = ""
        best_gain = -1

        for f in features:
            for c in class_opts:
                e = self.entropy(df, f, label, c)
                total_gain = set_entropy - e
                if (total_gain > best_gain):
                    best_gain = total_gain
                    best_feat = f

        return best_feat

    # classification prediction for a given row
    # will recursively traverse down node (tree) if it can
    #
    # arguments:
    #   - node: current node or subtree that is being evaluated for prediction
    #   - row: instance to be predicted for
    #
    # returns
    #   - the predicted class for the row
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

            # handle cases where the trained tree does not have a branch
            # in this scenario, we'll just end early at current node
            if (picked_child == None):
                return node.decision

            # if the chosen branch has been pruned, end at current node
            elif (picked_child.pruned):
                return node.decision

            # otherwise, continue recursively traversing down tree
            else:
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

        df = df.copy()
        df['right'] = df.apply(lambda row : check_if_right(tree, row), axis = 1)
        tot = df.shape[0]
        right = df[df['right'] == True].shape[0]
        return round(right / tot, 5)

    def prune_tree(self, tree, label):

        def prune_helper(node, label):
            if (node.children is None or len(node.children) == 0):
                accuracy = self.test_tree(tree, self.validation_set, label)
                node.pruned = True
                self.num_pruned_nodes += 1
                pruned_accuracy = self.test_tree(tree, self.validation_set, label)

                # accuracy dropped with pruning, so we'll keep this node
                if (pruned_accuracy < accuracy):
                    node.pruned = False
                    self.num_pruned_nodes -= 1

                return

            else:
                for c in node.children:
                    prune_helper(c, label)

                accuracy = self.test_tree(tree, self.validation_set, label)
                node.pruned = True
                self.num_pruned_nodes += 1
                pruned_accuracy = self.test_tree(tree, self.validation_set, label)
                if (pruned_accuracy < accuracy):
                    node.pruned = False
                    self.num_pruned_nodes -= 1

        prune_helper(tree, label)

    # method that builds up the id3 tree itself
    # recursively builds down tree as it splits the data
    # will stop under following conditions:
    #   - no more attributes to use
    #   - given feature only has 1 value option
    #   - only leaf nodes created, i.e. remaining rows are all the same class
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
    def id3_tree(self, df, label, tree = None, features = None, prior_value = None):
        # default root node
        root_node = Node(items = df, transition = prior_value)

        # initialize root node with actual feature
        root = self.pick_best_feature(df, label)
        root_node.feature = root
        
        # set the default decision for any given node to be most common class
        dec = self.most_common_class(df, label)
        root_node.decision = self.most_common_class(df, label)
        tree = root_node

        # if there are no more features to split on, then just return
        if (len(features) <= 1):
            return tree

        # there are more than 1 features left
        next_features = np.delete(features, np.where(features == [root]))
        feature_opts = self.feature_map[root]

        # make leaf node with decision
        if (len(feature_opts) == 1):
            print('only 1 feature option for ', feature_opts)
            return tree

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
                self.num_nodes += 1

            # otherwise, feature-attr pair will be recursively split
            # if the split for this feature option doesn't yield any children
            # then just return current tree
            else:
                subset = df_filtered
                subset = subset.drop(columns = root)
                if (subset.shape[0] > 1):
                    subtree = self.id3_tree(subset, label, tree, next_features, f)
                    tree.append_child(subtree)
                    self.num_nodes += 1
                else:
                    return tree

        return tree