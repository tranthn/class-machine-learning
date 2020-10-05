from math import log  # We need this to compute log base 2
from collections import Counter  # Used for counting

# File name: id3.py
# Author: Addison Sears-Collins
# Date created: 7/5/2019
# Python version: 3.7
# Description: Iterative Dichotomiser 3 algorithm

# Required Data Set Format for Classification Problems:
# Columns (0 through N)
# 0: Class
# 1: Attribute 1
# 2: Attribute 2
# 3: Attribute 3
# ...
# N: Attribute N

class Node:
   
  # Method used to initialize a new node's data fields with initial values
  def __init__(self, label):
 
    # Declaring variables specific to this node
    self.attribute = None  # Attribute (e.g. 'Outlook')
    self.attribute_values = []  # Values (e.g. 'Sunny')
    self.label = label   # Class label for the node (e.g. 'Play')
    self.children = {}   # Keeps track of the node's children
     
    # References to the parent node
    self.parent_attribute = None
    self.parent_attribute_value = None
 
    # Used for pruned trees
    self.pruned = False  # Is this tree pruned? 
    self.instances_labeled = []

def ID3(instances, default):
    """
    Parameters:
      instances: A list of dictionaries where each dictionary is an instance. 
                 Each dictionary contains attribute:value pairs 
                 e.g.: instances =
                   {'Class':'Play','Outlook':'Sunny','Temperature':'Hot'}
                   {'Class':'Don't Play','Outlook':'Rain','Temperature':'Cold'}
                   {'Class':'Play','Outlook':'Overcast','Temperature':'Hot'}
                   ...
                   etc.
                 The first attribute:value pair is the 
                 target variable (i.e. the class of that instance)
      default: The default class label (e.g. 'Play')
    Returns:
      tree: An object of the Node class
    """
    # The len method returns the number of items in the list
    # If there are no more instances left, return a leaf that is labeled with
    # the default class
    if len(instances) == 0:
        return Node(default)

    classes = []  # Create an empty list named 'classes'

    # For each instance in the list of instances, append the value of the class
    # to the end of the classes list
    for instance in instances:
        classes.append(instance['Class'])

    # If all instances have the same class label or there is only one instance
    # remaining, create a leaf node labeled with that class.
    # Counter(list) creates a tally of each element in the list. This tally is
    # represented as an element:tally pair.
    if len(Counter(classes)) == 1 or len(classes) == 1:
        tree = Node(mode_class(instances))
        return tree

    # Otherwise, find the best attribute, the attribute that maximizes the gain
    # ratio of the data set, to be the next decision node.
    else:
        # Find the name of the most informative attribute of the data set
        # e.g. "Outlook"
        best_attribute = most_informative_attribute(instances)

        # Initialize a tree with the most common class
        # e.g. "Play"
        tree = Node(mode_class(instances))

        # The most informative attribute becomes this decision node
        # e.g. "Outlook" becomes this node
        tree.attribute = best_attribute

        best_attribute_values = []

        # The branches of the node are the values of the best_attribute
        # e.g. "Sunny", "Overcast", "Rainy"
        # Go through each instance and create a list of the values of
        # best_attribute
        for instance in instances:
            try:
                best_attribute_values.append(instance[best_attribute])
            except:
                no_best_attribute = True
        # Create a list of the unique best attribute values
        # Set is like a list except it extracts nonduplicate (unique)
        # items from a list.
        # In short, we create a list of the set of unique
        # best attribute values.
        tree.attribute_values = list(set(best_attribute_values))

        # Now we need to split the instances. We will create separate subsets
        # for each best attribute value. These become the child nodes
        # i.e. "Sunny", "Overcast", "Rainy" subsets
        for best_attr_value_i in tree.attribute_values:

            # Generate the subset of instances
            instances_i = []
            # Go through one instance at a time
            for instance in instances:
                # e.g. If this instance has "Sunny" as its best attribute value
                if instance[best_attribute] == best_attr_value_i:
                    # Add this instance to the list
                    instances_i.append(instance)

            # Create a subtree recursively
            subtree = ID3(instances_i, mode_class(instances))

            # Initialize the values of the subtree
            subtree.instances_labeled = instances_i

            # Keep track of the state of the subtree's parent (i.e. tree)
            subtree.parent_attribute = best_attribute  # parent node
            subtree.parent_attribute_value = best_attr_value_i  # branch name

            # Assign the subtree to the appropriate branch
            tree.children[best_attr_value_i] = subtree

        # Return the decision tree
        return tree


def mode_class(instances):
    """
    Parameters: 
      instances: A list of dictionaries where each dictionary is an instance. 
        Each dictionary contains attribute:value pairs 
    Returns:
      Name of the most common class (e.g. 'Don't Play')
    """

    classes = []  # Create an empty list named 'classes'

    # For each instance in the list of instances, append the value of the class
    # to the end of the classes list
    for instance in instances:
        classes.append(instance['Class'])

    # The 1 ensures that we get the top most common class
    # The [0][0] ensures we get the name of the class label and not the tally
    # Return the name of the most common class of the instances
    return Counter(classes).most_common(1)[0][0]


def prior_entropy(instances):
    """
    Calculate the entropy of the data set with respect to the actual class
    prior to splitting the data.
    Parameters:
      instances: A list of dictionaries where each dictionary is an instance. 
        Each dictionary contains attribute:value pairs 
    Returns:
      Entropy value in bits
    """
    # For each instance in the list of instances, append the value of the class
    # to the end of the classes list
    classes = []  # Create an empty list named 'classes'

    for instance in instances:
        classes.append(instance['Class'])
    counter = Counter(classes)

    # If all instances have the same class, the entropy is 0
    if len(counter) == 1:
        return 0
    else:
        # Compute the weighted sum of the logs of the probabilities of each
        # possible outcome
        entropy = 0
        for c, count_of_c in counter.items():
            probability = count_of_c / len(classes)
            entropy += probability * (log(probability, 2))
        return -entropy


def entropy(instances, attribute, attribute_value):
    """
    Calculate the entropy for a subset of the data filtered by attribute value
    Parameters:
      instances: A list of dictionaries where each dictionary is an instance. 
        Each dictionary contains attribute:value pairs 
      attribute: The name of the attribute (e.g. 'Outlook')
      attribute_value: The value of the attribute (e.g. 'Sunny')
    Returns:
      Entropy value in bits
    """
    # For each instance in the list of instances, append the value of the class
    # to the end of the classes list
    classes = []  # Create an empty list named 'classes'

    for instance in instances:
        if instance[attribute] == attribute_value:
            classes.append(instance['Class'])
    counter = Counter(classes)

    # If all instances have the same class, the entropy is 0
    if len(counter) == 1:
        return 0
    else:
        # Compute the weighted sum of the logs of the probabilities of each
        # possible outcome
        entropy = 0
        for c, count_of_c in counter.items():
            probability = count_of_c / len(classes)
            entropy += probability * (log(probability, 2))
        return -entropy


def gain_ratio(instances, attribute):
    """
    Calculate the gain ratio if we were to split the data set based on the values
    of this attribute.
    Parameters:
      instances: A list of dictionaries where each dictionary is an instance. 
        Each dictionary contains attribute:value pairs 
      attribute: The name of the attribute (e.g. 'Outlook')
    Returns:
      The gain ratio
    """
    # Record the entropy of the combined set of instances
    priorentropy = prior_entropy(instances)

    values = []

    # Create a list of the attribute values for each instance
    for instance in instances:
        values.append(instance[attribute])
    # Store the frequency counts of each attribute value
    counter = Counter(values)

    # The remaining entropy if we were to split the instances based on this attribute
    # This is a weighted entropy score sum
    remaining_entropy = 0

    # This variable is used for the gain ratio calculation
    split_information = 0

    # items() method returns a list of all dictionary key-value pairs
    for attribute_value, attribute_value_count in counter.items():
        probability = attribute_value_count/len(values)
        remaining_entropy += (probability * entropy(
            instances, attribute, attribute_value))
        split_information += probability * (log(probability, 2))

    information_gain = priorentropy - remaining_entropy

    split_information = -split_information

    gainratio = None

    if split_information != 0:
        gainratio = information_gain / split_information
    else:
        gainratio = -1000

    return gainratio


def most_informative_attribute(instances):
    """
    Choose the attribute that provides the most information if you were to
    split the data set based on that attribute's values. This attribute is the 
    one that has the highest gain ratio.
    Parameters:
      instances: A list of dictionaries where each dictionary is an instance. 
        Each dictionary contains attribute:value pairs 
      attribute: The name of the attribute (e.g. 'Outlook')
    Returns:
      The name of the most informative attribute
    """
    selected_attribute = None
    max_gain_ratio = -1000

    # instances[0].items() extracts the first instance in instances
    # for key, value iterates through each key-value pair in the first
    # instance in instances
    # In short, this code creates a list of the attribute names
    attributes = [key for key, value in instances[0].items()]
    # Remove the "Class" attribute name from the list
    attributes.remove('Class')

    # For every attribute in the list of attributes
    for attribute in attributes:
        # Calculate the gain ratio and store that value
        gain = gain_ratio(instances, attribute)

        # If we have a new most informative attribute
        if gain > max_gain_ratio:
            max_gain_ratio = gain
            selected_attribute = attribute

    return selected_attribute


def accuracy(trained_tree, test_instances):
    """
    Parameters:
        trained_tree: A tree that has already been trained
        test_instances: A set of test instances
    Returns:
        Classification accuracy (# of correct predictions/# of predictions) 
    """
    # Set the counter to 0
    no_of_correct_predictions = 0

    for test_instance in test_instances:
        if predict(trained_tree, test_instance) == test_instance['Class']:
            no_of_correct_predictions += 1

    return no_of_correct_predictions / len(test_instances)


def predict(node, test_instance):
    '''
    Parameters:
        node: A trained tree node
        test_instance: A single test instance
    Returns:
        Class value (e.g. "Play")
    '''
    # Stopping case for the recursive call.
    # If this is a leaf node (i.e. has no children)
    if len(node.children) == 0:
        return node.label
    # Otherwise, we are not yet on a leaf node.
    # Call predict method recursively until we get to a leaf node.
    else:
        # Extract the attribute name (e.g. "Outlook") from the node.
        # Record the value of the attribute for this test instance into
        # attribute_value (e.g. "Sunny")
        attribute_value = test_instance[node.attribute]

        # Follow the branch for this attribute value assuming we have
        # an unpruned tree.
        if attribute_value in node.children and node.children[
                attribute_value].pruned == False:
            # Recursive call
            return predict(node.children[attribute_value], test_instance)

        # Otherwise, return the most common class
        # return the mode label of examples with other attribute values for the current attribute
        else:
            instances = []
            for attr_value in node.attribute_values:
                instances += node.children[attr_value].instances_labeled
            return mode_class(instances)


TREE = None


def prune(node, val_instances):
    """
    Prune the tree recursively, starting from the leaves
    Parameters:
        node: A tree that has already been trained
        val_instances: The validation set        
    """
    global TREE
    TREE = node

    def prune_node(node, val_instances):
        # If this is a leaf node
        if len(node.children) == 0:
            accuracy_before_pruning = accuracy(TREE, val_instances)
            node.pruned = True

            # If no improvement in accuracy, no pruning
            if accuracy_before_pruning >= accuracy(TREE, val_instances):
                node.pruned = False
            return

        for value, child_node in node.children.items():
            prune_node(child_node, val_instances)

        # Prune when we reach the end of the recursion
        accuracy_before_pruning = accuracy(TREE, val_instances)
        node.pruned = True

        if accuracy_before_pruning >= accuracy(TREE, val_instances):
            node.pruned = False

    prune_node(TREE, val_instances)
