#!/usr/bin/env python3
import sys
import numpy as np
from termcolor import colored, cprint

"""
    – S – This square is on the starting line.
    – F – This square is on the finish line.
    – . – This square is open racetrack.
    – # – This square is off the racetrack (i.e., a wall).
"""

class TrackSimulator():
    def __init__(self, track = None, start_pos = (0, 0), start_velocity = (0, 0)):
        self.track = track
        self.position = start_pos
        self.velocity = start_velocity

        # array of coordinates to indicate path taken (for printing)
        self.path = []

    # helper that returns coordinates of a given feature in track (S, F, ./track, #/wall)
    def _find_coordinate(self, char = ''):
        for coords, value in np.ndenumerate(self.track):
            if (value == char):
                return coords
        return (-1, -1)

    def pretty_print(self):
        one_spacer = ' '
        two_spacer = '  '
        three_spacer = '   '
        r = 0

        # print column headers, with spacers to handle single vs. double-digit numbers
        print('\t   ', end = '')
        for i in range(0, self.track.shape[1]):
            spacer = one_spacer if i > 9 else two_spacer
            cprint('{0}{1}'.format(i, spacer), 'red', end = '')

        for coords, value in np.ndenumerate(self.track):
            # print row headers
            if (coords == (0, 0) or r < coords[0]):
                spacer = one_spacer if coords[0] > 9 else two_spacer
                print()
                cprint('\t{0}{1}'.format(coords[0], spacer), 'red', end = '')

            # if we're on current position or on path, print with color for visual indication
            spacer = one_spacer if coords[0] > 9 else two_spacer 
            if (self.position == coords or (coords[0], coords[1]) in self.path):
                cprint(value + spacer, 'green', end = '')
            else:
                print(value + spacer, end = '')

            r = coords[0]
        
        print('\n')
    

####################################################################################
    def initialize_track(self):
        self.position = self._find_coordinate('S')

    def move(self, velocity):
        p1 = self.position[0]
        p2 = self.position[1]

        self.position = (velocity[0] + p1, velocity[1] + p2)

        # should check if it collides into wall, but assume predetermined path for now
        self.path.append(self.position)

    def accelerate(self, x, y):
        self.velocity[0] += x
        self.velocity[1] += y

    ## helper that runs through with predetermined test path to finish line
    ## track is initialized on a start position already
    def test_run(self):
        moves = [
            (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),
            (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),  (0, 1), (0, 1),
            (-1, 0), (-1, 0), (-1, 0), (-1, 0),  (-1, 0)
        ]

        for m in moves:
            self.move(m)