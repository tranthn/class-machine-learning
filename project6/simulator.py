#!/usr/bin/env python3
import numpy as np
import random
from termcolor import colored, cprint

"""
    – S – This square is on the starting line.
    – F – This square is on the finish line.
    – . – This square is open racetrack.
    – # – This square is off the racetrack (i.e., a wall).
"""

class TrackSimulator():
    def __init__(self, track = None, min_velocity = -5, max_velocity = 5, crash_restart = False):
        self.track = track
        self.crash_restart = False
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

        # position = r, c 
        # r = row, c = column to match dataframe/matrix representation
        self.position = (0,0)
        self.velocity = (0,0)

        # array of coordinates to indicate path taken (for printing)
        self.path = []

    # helper that returns coordinates of a given feature in track (S, F, ./track, #/wall)
    def _find_coordinate(self, char = ''):
        for coords, value in np.ndenumerate(self.track):
            if (value == char):
                return coords
        return (-1, -1)

    def nrows(self):
        return self.track.shape[0]

    def ncols(self):
        return self.track.shape[1]

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

            spacer = two_spacer 

            # if we're on current position or on path, print with color for visual indication
            if (self.position == coords):
                cprint('X' + spacer, 'green', end = '')
            else:
                print(value + spacer, end = '')

            r = coords[0]
        
        print('\n')
    
####################################################################################
####################################################################################

    # helper that returns all open positions within racetrack
    def get_all_points_of(self, target = ''):
        targets = []
        for coords, value in np.ndenumerate(self.track):
            if (value == target):
                targets.append(coords)

        return targets

    # checks if given position hits a wall or goes out of bounds
    def boundary_check(self, position):
        r = position[0]
        c = position[1]
        rows = self.track.shape[0]
        cols = self.track.shape[1]

        # check for out of bounds coordinates
        if (r < 0 or c < 0 or r >= rows or c >= cols):
            # cprint('out of bounds', 'magenta')
            return False

        # reverse y, x since y indicates row position, while x indicates column position
        elif (self.track[r, c] == '#'):
            # cprint('hit a wall', 'magenta')
            return False
        else:
            return True

    """
        two handling strategies
            1 - move to x,y closest to crash point
            2 - move to start
    """
    def get_restart_position(self, crash_site):
        if not self.crash_restart:
            track_pts = self.get_all_points_of('.')
            start_pts = self.get_all_points_of('S')
            open_track = track_pts.append(start_pts)

            nrows = self.nrows()
            ncols = self.ncols()
            vr = self.velocity[0]
            vc = self.velocity[1]
            cr = crash_site[0]
            cc = crash_site[1]

            search_radius = max(nrows, ncols)
            # cprint('crash {0}'.format(crash_site), 'red')

            ## scenarios:
            #   - if trajectory was negative, we search radius in (0, positive direction)
            #   - if trajectory was 0, we search radius in both negative -> positive radius
            #   - if trajectory was positive, we search radius (-radius, 0)
            for radius in range(search_radius):
                if (vr < 0):
                    row_range = range(0, radius)
                elif (vr == 0):
                    row_range = range(-radius, radius)
                else:
                    row_range = range(-radius, 1)

                for row_offset in row_range:
                    r = cr + row_offset
                    c_radius = radius - abs(row_offset)

                    # c_radius is constrained by row radius
                    # we follow similar trajectory handling to above
                    if (vc < 0):
                        column_range = range(0, cc + c_radius)
                    elif (vc == 0):
                        column_range = range(cc - c_radius, cc + c_radius)
                    else:
                        column_range = range(cc - c_radius, cc)

                    for c in column_range:
                        if (r < 0 or c < 0 or r >= nrows or c >= ncols):
                            continue

                        if (r, c) in track_pts:
                            return(r, c)

            return self.position
        else:
            return self.start_pos

    # gives next position based off current position and velocity
    def get_next_position(self):
        v = self.velocity
        p1 = self.position[0]
        p2 = self.position[1]
        position = (self.velocity[0] + self.position[0],
                    self.velocity[1] + self.position[1])

        return position

    """
        Attempts to accelerate, i.e. adjust the velocity.
        Has 80% of succeeding and adjusts velocity within [-5, 5] limit

        arguments:
            - rise: the y-value or which row we adjust to within the 2d track array
            - run: the x-value or which column we adjust to within the 2d track array
    """
    def accelerate(self, rise, run):
        percent = round(random.random() * 100, 0)
        if (percent > 20):  
            vr = self.velocity[0] + rise
            vc = self.velocity[1] + run

            # ensure velocity doesn't go below min velocity or above max velocity
            vr2 = max(vr, self.min_velocity) if (vr < 0) else min(vr, self.max_velocity)
            vc2 = max(vc, self.min_velocity) if (vc < 0) else min(vc, self.max_velocity)
            self.velocity = (vr2, vc2)
            # print('v2', self.velocity)
        # else:
            # cprint('accelerate failed', 'red')
            # print('v1', self.velocity)

        return self.velocity

    # method tries to adjust position, checking boundaries first
    # if new position goes off track, we will restart position
    def move(self):
        position = self.get_next_position()
        if (self.boundary_check(position)):
            self.temp_position = position
        else:
            self.velocity = [0,0]
            self.temp_position = self.get_restart_position(position)

        return self.temp_position

    # sets position officially
    def finalize_move(self):
        self.position = self.temp_position

    def run_trial(self, policy):
        # stop racing after 300 moves, all tracks have < 300 open spots
        # just to prevent excessive runs early in learning process
        stop_after = 300
        self.position = self._find_coordinate('S')
        moves = 1

        for moves in range(stop_after):
            r = self.position[0]
            c = self.position[1]
            vr = self.velocity[0]
            vc = self.velocity[1]

            if self.track[r,c] == 'F':
                break

            next_action = policy[(r, c, vr, vc)]
            
            # accelerate and move
            self.accelerate(next_action[0], next_action[1])
            self.move()
            self.finalize_move()

        return moves

    ## helper that runs through with predetermined test path to finish line
    ## track is initialized on a start position already
    def test_run(self):
        self.velocity = [-1, 1]
        self.accelerate(0, 0) # rise, run
        self.move()
        self.finalize_move()