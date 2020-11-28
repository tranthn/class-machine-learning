#!/usr/bin/env python3
import sys
import numpy as np

class Learner():

    def __init__(self, learning_algorithm = 'q'):
        self.algo = learning_algorithm
        self.track = track

    # sequence of V_t using Q_t(s, a)
    #   - Q is auxillary function, represents estimate of value of action *a* taken on state *s*
    #   - values of Q_t and V_t updated as we go along, until we drop below error threshold
    #
    # value iteration is essentially a form of dynamic programming
    def value_iteration(self):
        return None
    
    # model free, we do not know state transition probabilities or rewards
    # goal is to learn value-function Q
    # uses epsilon to determine how greedy (= 0) or optimal (increased)
    #
    # agent is run in environment as trial, state-action pairs collected
    #   - Q_t(s, a): estimate value for (s, a) pair
    #   - alpha_t(s, a) in [0, 1]: learning rate at given time *t*
    #   - gamma in [0, 1]: discount factor
    #   - s': successor state
    #
    # updates based on what would have been optimal according to current estimate of our Q function
    # selects next option using softmax to pick optimal choice
    def q_learning(self):
        return None
    
    # updates based on the immediately experienced / preceding Q-value
    def sarsa(self):
        return None