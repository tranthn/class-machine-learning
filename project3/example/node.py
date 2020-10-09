#!/usr/bin/env python3

# File name: node.py
# Author: Addison Sears-Collins
# Date created: 7/6/2019
# Python version: 3.7
# Description: Used for constructing nodes for a tree
 
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