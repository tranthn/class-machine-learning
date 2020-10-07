#!/usr/bin/env python3
import sys
import math
import numpy as np
import pandas as pd

# binary split for numeric columns
#
# returns
#   - median point
def get_numeric_split(df, column):
    print('\n- numeric split -')
    print('col\t', column)
    sorted = df.sort_values(by = column)
    md1 = int(df.shape[0] / 2)
    md2 = int(md1 + 1)

    values = sorted[column].to_numpy()
    split = (values[md1] + values[md2]) / 2
    print('split\t', split)

    med = df[column].median()
    print('median\t', med)

    return med

# calculate individual information gain value for given position / total combo
#
# arguments
#   - total: total # items in a given feature-attr combo
#   - pos: # of class positive examples with given feature-attr combo
#
# returns
#   - calculated log2 information gain
def info_gain(total, pos):
    # negative count is just remainder
    neg = total - pos
    p_pn = pos / total
    n_pn = neg / total

    if (p_pn == 0 or n_pn == 0):
        return 0
    else:
        i = (-p_pn * math.log(p_pn, 2)) - (n_pn * math.log(n_pn, 2))
        return i

def is_categorical(df, feature):
    ftype = df[feature].dtype
    categorical = ftype != 'float64' and ftype != 'int64'
    return categorical

def build_query_string(features, operators, values):
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
def entropy(df, feature, class_label):
    tot = df.shape[0]
    ent = 0

    if (is_categorical(df, feature)):        
        # find unique value options for this feature column
        feature_opts = df[feature].unique()

        for f in feature_opts:
            # build query string, e.g. "feature == f"
            # query will filter dataframe based on the query condition
            qry = build_query_string([feature], ['=='], [f])
            df_grouped = df.query(qry)

            # determine how many classes represented after filtering above
            class_count = len(df_grouped[class_label].unique())

            # this is hard-coded for class belonging / class == 1
            df_grouped_cl = df_grouped[df_grouped[class_label] == 1]
            
            # check if the feature has a split for predicting class or not
            # if the length = 0, that means this feature-opt combo all fell
            # within the same outcome class, take else branch
            if (class_count > 1):
                p = df_grouped_cl.shape[0]
                pn = df_grouped.shape[0]
                if_gain = info_gain(pn, p)
                ent += (pn / tot) * if_gain
            else:
                # gain should be 0
                if_gain = 0
    
    # handle numeric columns with separate split logic
    else:      
        split_point = get_numeric_split(df, feature)
        q_lte = build_query_string([feature], ['<='], [split_point])
        q_gt = build_query_string([feature], ['>'], [split_point])
        feature_opts = [q_lte, q_gt]

        for f in feature_opts:
            # query split
            split = df.query(f)
            split_tot = split.shape[0]
            split_positive = split[split[class_label] == 1].shape[0]

            print('# <= {0}:\t {1}'.format(split_point, split_tot))
            print('# <= {0} (p):\t{1} '.format(split_point, split_positive))

            split_gain = (split_tot / tot) * info_gain(split_tot, split_positive)

            ent += split_gain

    return ent

# find best feature to be root node
def pick_feature(df, class_label):
    tot = df.shape[0]
    cols = df.columns.drop(class_label)

    # calculate set entropy
    grouping = df.groupby(by = [class_label]).size()
    set_entropy = info_gain(tot, grouping[1])
    best_feat = ""
    best_gain = -1

    for f in cols:
        e = entropy(df, f, class_label)
        total_gain = set_entropy - e
        if (total_gain > best_gain):
            best_gain = total_gain
            best_feat = f

    return best_feat

def id3_tree(df, label, tree = None, features = None, prior_value = None):
    print('---------------------\n')

    # default root node
    root_node = Node(items = df, transition_value = prior_value)

    # if tree hasn't started yet, set to current root
    if (tree == None):
        tree = root_node

    # label decision on node
    if (len(features) <= 1):
        return tree

    # initialize root node with actual feature
    root = pick_feature(df, label)
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
    print('\nnext root', root)
    next_features = np.delete(features, np.where(features == [root]))

    for f in feature_opts.values:
        # only 1 class represented, so it becomes leaf
        nested_grouping = df.groupby(by = [root, label], dropna = False).size()

        # feature-attr pair only represents 1 class, so it'll become leaf
        if (len(nested_grouping[f]) == 1):
            # get the only represented class
            dec = nested_grouping[f].index.values[0]
            leaf = Node(feature = root, transition_value = f, decision = dec)
            tree.append_child(leaf)

        # otherwise, feature-attr pair will be recursively split
        else:
            subset = df[df[root] == f]
            subset = subset.drop(columns = root)
            subtree = id3_tree(subset, label, tree, next_features, f)
            tree.append_child(subtree)

    return tree

######################################################
class Node():
    def __init__(self, feature = None, transition_value = None, split_fn = None, items = None, decision = None):
        self.feature = feature
        self.split_fn = split_fn
        self.children = list()
        self.decision = decision
        self.transition_value = transition_value
        self.items = items

    def print(self, levels = 0):
        pre = '\t' * levels
        print()
        print('{0} feat: {1}'.format(pre, self.feature))
        print('{0} transition value: {1}'.format(pre, self.transition_value))
        if not (self.items is None):
            print('{0} items #: {1}'.format(pre, len(self.items)))

        if (len(self.children) > 0):
            print('{0} children #: {1}'.format(pre, len(self.children)))
            for c in self.children:
                c.print(levels = levels + 1)
        else:
            print(pre, 'leaf node')
            print(pre, 'decision = {0}'.format(self.decision))

    def append_child(self, node):
        self.children.append(node)