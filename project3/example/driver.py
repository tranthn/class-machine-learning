#!/usr/bin/env python3
import id3
import reader
import random
import fold
from matplotlib import pyplot as plt
 
 
# File name: results.py
# Author: Addison Sears-Collins
# Date created: 7/6/2019
# Python version: 3.7
# Description: Results of the Iterative Dichotomiser 3 runs
# This source code is the driver for the entire program
 
# Required Data Set Format for Classification Problems:
# Columns (0 through N)
# 0: Class
# 1: Attribute 1 
# 2: Attribute 2
# 3: Attribute 3 
# ...
# N: Attribute N

ALGORITHM_NAME = "Iterative Dichotomiser 3"
 
def main():
 
    print("Welcome to the " +  ALGORITHM_NAME + " Program!")
    print()
 
    # Enter the name of your input file
    #file_name = 'car.txt'
    file_name = input("Enter the name of your input file (e.g. car.txt): ") 
     
 
    # Show functioning of the program
    #trace_runs_file = 'car_id3_trace_runs.txt'
    trace_runs_file = input(
       "Enter the name of your trace runs file (e.g. car_id3_trace_runs.txt): ")     
 
    # Save the output graph of the results
    #imagefile = 'car_id3_results.png'
    # imagefile = input(
    #     "Enter the name of the graphed results file (e.g. foo.png): ")     
 
    # Open a new file to save trace runs
    outfile_tr = open(trace_runs_file,"w") 
 
    outfile_tr.write("Welcome to the " +  ALGORITHM_NAME + " Program!" + "\n")
    outfile_tr.write("\n")
 
    data = reader.parse(file_name)
    pruned_accuracies_avgs = []
    unpruned_accuracies_avgs = []
 
    # Shuffle the data randomly
    random.shuffle(data)
 
    # This variable is used for the final graph. Places
    # upper limit on the x-axis.
    # 10% of is pulled out for the validation set
    # 20% of that set is used for testing in the five-fold
    # stratified cross-validation
    # Round up to the nearest value of 10
    upper_limit = (round(len(data) * 0.9 * 0.8) - round(
        len(data) * 0.9 * 0.8) % 10) + 10
    #print(str(upper_limit)) # Use for debugging
    if upper_limit <= 10:
        upper_limit = 50
 
    # Get the most common class in the data set.
    default = id3.mode_class(data)
 
    # Pull out 10% of the data to be used as a validation set
    # The remaining 90% of the data is used for cross validation.
    validation_set = data[: 1*len(data)//10]
    data = data[1*len(data)//10 : len(data)]
 
    # Generate the five stratified folds
    fold0, fold1, fold2, fold3, fold4 = fold.get_five_folds(
        data)
 
    # Generate lists to hold the training and test sets for each experiment
    testset = []
    trainset = []
 
    # Create the training and test sets for each experiment
    # Experiment 0
    testset.append(fold0)
    trainset.append(fold1 + fold2 + fold3 + fold4)
 
    # Experiment 1
    testset.append(fold1)
    trainset.append(fold0 + fold2 + fold3 + fold4)
 
    # Experiment 2
    testset.append(fold2)
    trainset.append(fold0 + fold1 + fold3 + fold4)
 
    # Experiment 3
    testset.append(fold3)
    trainset.append(fold0 + fold1 + fold2 + fold4)
     
    # Experiment 4
    testset.append(fold4)
    trainset.append(fold0 + fold1 + fold2 + fold3)
 
    step_size = len(trainset[0])//20
 
    for length in range(10, upper_limit, step_size):
        print('Number of Training Instances:', length)
        outfile_tr.write('Number of Training Instances:' + str(length) +"\n")
        pruned_accuracies = []
        unpruned_accuracies = []
 
        # Run all 5 experiments for 5-fold stratified cross-validation
        for experiment in range(1):
 
            # Each experiment has a training and testing set that have been 
            # preassigned.
            train = trainset[experiment][: length]
            test = testset[experiment]
 
            # Pruned
            tree = id3.ID3(train, default)
            id3.prune(tree, validation_set)
            acc = id3.accuracy(tree, test)
            pruned_accuracies.append(acc)
 
            # Unpruned
            tree = id3.ID3(train, default)
            acc = id3.accuracy(tree, test)
            unpruned_accuracies.append(acc) 
         
        # Calculate and store the average classification 
        # accuracies for each experiment
        avg_pruned_accuracies = sum(pruned_accuracies) / len(pruned_accuracies)
        avg_unpruned_accuracies = sum(unpruned_accuracies) / len(unpruned_accuracies)
 
        print("Classification Accuracy for Pruned Tree:", avg_pruned_accuracies) 
        print("Classification Accuracy for Unpruned Tree:", avg_unpruned_accuracies)
        print()
        outfile_tr.write("Classification Accuracy for Pruned Tree:" + str(
            avg_pruned_accuracies) + "\n") 
        outfile_tr.write("Classification Accuracy for Unpruned Tree:" + str(
                avg_unpruned_accuracies) +"\n\n")
 
        # Record the accuracies, so we can plot them later
        pruned_accuracies_avgs.append(avg_pruned_accuracies)
        unpruned_accuracies_avgs.append(avg_unpruned_accuracies) 
     
    # Close the file
    outfile_tr.close()
 
    # plt.plot(range(10, upper_limit, step_size), pruned_accuracies_avgs, label='pruned tree')
    # plt.plot(range(10, upper_limit, step_size), unpruned_accuracies_avgs, label='unpruned tree')
    # plt.xlabel('Number of Training Instances')
    # plt.ylabel('Classification Accuracy on Test Instances')
    # plt.grid(True)
    # plt.title("Learning Curve for " +  str(file_name))
    # plt.legend()
    # plt.savefig(imagefile) 
    # plt.show()
     
 
main()