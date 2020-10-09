#!/usr/bin/env python3
import csv # Library to handle csv-formatted files
 
# File name: parse.py
# Author: Addison Sears-Collins
# Date created: 7/6/2019
# Python version: 3.7
# Description: Used for parsing the input data file
 
def parse(filename):
    """
    Parameters: 
        filename: Name of a file
    Returns: 
        data: Information on the attributes and the data as a list of 
              dictionaries. Each instance is a different dictionary.
    """
 
    # Initialize an empty list named 'data'
    data = []
 
    # Open the file in READ mode.
    # The file object is named 'file'
    with open(filename, 'r') as file:
 
        # Convert the file object named file to a csv.reader object. Save the
        # csv.reader object as csv_file
        csv_file = csv.reader(file)
 
        # Return the current row (first row) and advance the iterator to the
        # next row. Since the first row contains the attribute names (headers),
        # save them in a list called headers
        headers = next(csv_file)
 
        # Extract each of the remaining data rows one row at a time
        for row in csv_file:
            # append method appends an element to the end of the list
            # The element that is appended is a dictionary.
            # A dictionary contains search key-value pairs, analogous to
            # word-definition in a regular dictionary.
            # In this case, each instance is a separate dictionary.
            # The zip method joins two lists together so that we have
            # attributename(header)-value(row) pairs for each instance
            # in the data set
            data.append(dict(zip(headers, row)))
 
    return data
 
##Used for debugging
#name_of_file =  "abalone.txt" 
#data = parse(name_of_file)
#print(*data, sep = "\n")
#print()
#print(str(len(data)))