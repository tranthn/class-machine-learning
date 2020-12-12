#!/usr/bin/env python3
import time
import json
import random
import data_loader as dl
from simulator import TrackSimulator
from q_learner import Q_SARSA_Learner
from value_iteration import ValueIteration

# helper to write out the policy dictionary structure to a file
def write_output_helper(file_prefix, input):
    # timestamp to append to file, month-day_hour
    timestamp = time.strftime('%m-%d_%H', time.localtime())
    filename = 'out/' + file_prefix + '.out'
    with open(filename, 'w') as f:
        for k,v in input.items():
            print("{0}:{1}".format(k,v), file = f)

"""
    loads policy dictionary structure from file
    file is structure as one line per state-action combo like so:
        (0, 0, 3, 1):(-1, -1)

    allows re-running trials without retraining
"""
def load_policy(file):
    policy = {}
    with open(file) as f:
        for line in f:
            # print(line)
            parts = line.split(':')
            key = eval(parts[0])
            val = eval(parts[1])
            policy[key] = val
    
    return policy


"""
    helper that trains and runs trials given a track and learner instance
        - prints out policy finalization time, no. moves taken and trial runtime

    arguments:
        - simulator: TrackSimulator instance loaded with a track
        - learner: instance of ValueIteration or Q_SARSA_Learner, with hyperparameters present
        - iterations: number of training cycles to run with learner
        - trial_runs: number of time trials to run
        - policy: if policy was loaded, it will be passed in here, otherwise None
"""
def trial_helper(simulator, learner, iterations, trial_runs, trackname, policy):
    print('\nstart of training, for {0} total iterations'.format(iterations))
    st = time.time()
    if policy is None:
        policy = learner.train(iterations = iterations)
    else:
        print('using loaded policy')

    print('\npolicy finalized after {:.2f}s'.format(time.time() - st))
    print('\n# time trials to run: ', trial_runs)
    print()
    for i in range(trial_runs):
        start_time = time.time()
        print('\ttrial', i)
        steps = simulator.run_trial(policy)        

        # store policy for later use if needed
        write_output_helper(trackname, policy)
        print('\t# moves taken: ', steps)
        elapsed = time.time() - start_time
        print('\ttrial runtime:\t{:.2f}s'.format(elapsed))
        print()

    print('=' * 100)

################################################################################
actions = [
    (-1,-1), (0,-1), (1,-1),
    (-1, 0), (0, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
]

vl_opts = [0, 1,  2, 3, 4, 5, -5, -4, -3, -2, -1]

# tiny test track
################################################################################
track = dl.load_tinytrack()
simulator = TrackSimulator(track = track, min_velocity = min(vl_opts), max_velocity = max(vl_opts), crash_restart = False)

learner = ValueIteration(env = simulator, vl_opts = vl_opts, actions = actions,
                        gamma = 1.0, epsilon = 0.01)

learner = Q_SARSA_Learner(env = simulator, vl_opts = vl_opts,
                    actions = actions, alpha = 0.25, gamma = 0.9)

learner = Q_SARSA_Learner(env = simulator, vl_opts = vl_opts, actions = actions,
                    alpha = 0.25, gamma = 0.9, sarsa = True)

# simulator.pretty_print()
# trial_helper(simulator, learner, 100000, 10, 'tinytrack', policy = None)

# l-track
################################################################################
track = dl.load_l()
print()
print('=' * 100)
print('RUNNING L TRACK')
simulator = TrackSimulator(track = track, min_velocity = min(vl_opts), max_velocity = max(vl_opts), crash_restart = False)
learner = ValueIteration(env = simulator, vl_opts = vl_opts, actions = actions,
                        gamma = 1.0, epsilon = 0.001)

qlearner = Q_SARSA_Learner(env = simulator, vl_opts = vl_opts,
                    actions = actions, alpha = 0.25, gamma = 0.9)

sarsalearner = Q_SARSA_Learner(env = simulator, vl_opts = vl_opts, actions = actions,
                    alpha = 0.25, gamma = 0.9, sarsa = True)

simulator.pretty_print()
policy = None
# policy = load_policy('out/L-track.out')

# print('=' * 100)
# print('-- Value Learning --')
# trial_helper(simulator, learner, 50, 10, 'L-track-val', policy = policy)

# print('=' * 100)
# print('-- Q-learning --')
# trial_helper(simulator, qlearner, 500000, 10, 'L-track-q', policy = policy)

# print('=' * 100)
# print('-- SARSA --')
# trial_helper(simulator, sarsalearner, 500000, 10, 'L-track-sarsa', policy = policy)

# r-track
################################################################################
track = dl.load_r()
print()
print('=' * 100)
print('RUNNING R TRACK')
simulator = TrackSimulator(track = track, min_velocity = min(vl_opts), max_velocity = max(vl_opts), crash_restart = True)

learner = ValueIteration(env = simulator, vl_opts = vl_opts, actions = actions,
                        gamma = 1.0, epsilon = 0.001)

qlearner = Q_SARSA_Learner(env = simulator, vl_opts = vl_opts,
                    actions = actions, alpha = 0.25, gamma = 0.9)

sarsalearner = Q_SARSA_Learner(env = simulator, vl_opts = vl_opts, actions = actions, 
                        alpha = 0.25, gamma = 0.9, sarsa = True)

simulator.pretty_print()
policy = None

# print('=' * 100)
# print('-- Value Learning --')
# trial_helper(simulator, learner, 50, 10, 'R-track-val', policy = policy)

# print('=' * 100)
# print('-- Q-learning --')
# trial_helper(simulator, qlearner, 500000, 10, 'R-track-q', policy = policy)

# print('=' * 100)
# print('-- SARSA --')
# trial_helper(simulator, sarsalearner, 500000, 10, 'R-track-sarsa', policy = policy)
# trial_helper(simulator, sarsalearner, 1000000, 10, 'R-track-sarsa', policy = policy)

# o-track
################################################################################
track = dl.load_o()
print()
print('=' * 100)
print('RUNNING O TRACK')
simulator = TrackSimulator(track = track, min_velocity = min(vl_opts), max_velocity = max(vl_opts), crash_restart = False)

learner = ValueIteration(env = simulator, vl_opts = vl_opts, actions = actions,
                        gamma = 1.0, epsilon = 0.001)

qlearner = Q_SARSA_Learner(env = simulator, vl_opts = vl_opts,
                    actions = actions, alpha = 0.25, gamma = 0.9)

sarsalearner = Q_SARSA_Learner(env = simulator, vl_opts = vl_opts, actions = actions,
                    alpha = 0.5, gamma = 0.9, sarsa = True)

simulator.pretty_print()
policy = None

# print('=' * 100)
# print('-- Value Learning --')
# trial_helper(simulator, learner, 50, 10, 'O-track-val', policy = policy)

# print('=' * 100)
# print('-- Q-learning --')
# trial_helper(simulator, qlearner, 500000, 10, 'O-track-q', policy = policy)
# print('=' * 100)

# print('-- SARSA --')
# trial_helper(simulator, sarsalearner, 500000, 10, 'O-track-sarsa', policy = policy)
# trial_helper(simulator, sarsalearner, 1000000, 10, 'O-track-sarsa', policy = policy)