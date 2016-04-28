"""Reinforcement learning for Tic Tac Toe as in
https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node10.html"""

import sys
import numpy as np
import pickle as pkl

BOARD_SIZE = 3

class Board(object):
    EMPTY = 0
    CROSS = 1
    CIRCL = 2
    TIE = 3

    def __init__(self):
        # first index is x, second index is y
        self.state = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype='I')
        self.winner = None

    def clear(self):
        self.__init__()

    def check_coord_in_range(self, coord):
        assert coord[0] < BOARD_SIZE
        assert coord[1] < BOARD_SIZE

    def check_coord_free(self, coord):
        assert self.state[coord[0], coord[1]] == self.EMPTY

    def set_cross(self, coord):
        """Set a cross at tuple coord"""
        self.check_coord_in_range(coord)
        self.check_coord_free(coord)
        self.state[coord[0], coord[1]] = self.CROSS

    def set_circl(self, coord):
        """Set a circle at tuple coord"""
        self.check_coord_in_range(coord)
        self.check_coord_free(coord)
        self.state[coord[0], coord[1]] = self.CIRCL

    def is_done(self):
        # check rows
        for y in range(BOARD_SIZE):
            unique = np.unique(self.state[y,])
            if len(unique) == 1 and unique[0] != self.EMPTY:
                self.winner = unique[0]
                return True
        # check columns
        for x in range(BOARD_SIZE):
            unique = np.unique(self.state[:, x])
            if len(unique) == 1 and unique[0] != self.EMPTY:
                self.winner = unique[0]
                return True
        # check diagonal
        unique = np.unique(np.diag(self.state))
        if len(unique) == 1 and unique[0] != self.EMPTY:
            self.winner = unique[0]
            return True
        # check anti-diagonal
        bs = BOARD_SIZE
        val = []
        for i in range(bs):
            val.append(self.state[bs-i-1, i])
        unique = np.unique(val)
        if len(unique) == 1 and unique[0] != self.EMPTY:
            self.winner = unique[0]
            return True
        # check if tie
        if len(self.state[self.state == self.EMPTY]) == 0:
            self.winner = self.TIE
            return True

        return False

    def who_won(self):
        return self.winner

    def get_state_string(self):
        return self.state.tostring()

    def get_empty_fields(self):
        return self.state.flat == self.EMPTY

    def convert_to_icon(self, value):
        if value == self.EMPTY:
            return ' '
        elif value == self.CROSS:
            return 'X'
        elif value == self.CIRCL:
            return 'O'

    def print_board(self):
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                sys.stdout.write(self.convert_to_icon(self.state[x, y]))
                if x < BOARD_SIZE-1:
                    sys.stdout.write(" | ")
            if y < BOARD_SIZE-1:
                sys.stdout.write("\n---------")
            sys.stdout.write("\n")

class Agent(object):
    STARTING_VALUE = 0.5
    RATE = 0.1
    EXPLORE_STEPS = 3
    def __init__(self, filename):
        self.state = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype='I').tostring()
        self.steps = 1

        if not filename:
            print "Starting with new value map"
            filename = 'value_map.pkl'
            self.values = { self.state: \
                        np.ones((BOARD_SIZE, BOARD_SIZE))*self.STARTING_VALUE}
        else:
            self.values = pkl.load(open(filename, 'r'))
            print "Loaded value map with", len(self.values), "states"

        self.filename = filename

    def update_state(self, next_values):
        value_diff = next_values - self.values[self.state]
        self.values[self.state] += self.RATE * value_diff

        print "Current value map:", self.values[self.state]

    def set_state(self, state):
        """Update the agent's internal board state and trigger a TD learning
        update"""

        if not state in self.values:
            self.values[state] = np.ones((BOARD_SIZE, BOARD_SIZE)) \
                                 * self.STARTING_VALUE
        print "Number of known states:", len(self.values)

        self.update_state(self.values[state])

        self.state = state


    def get_action(self, empty):
        """Choose the action with the highest probability"""
        print "Empty mask:", empty
        print "Filtered map:", self.values[self.state].flat[empty]
        masked_values = self.values[self.state].flat[empty]
        if self.steps % self.EXPLORE_STEPS == 0:
            flat_action_index = np.random.randint(0, len(masked_values))
        else:
            flat_action_index = np.argmax(masked_values)
        self.steps += 1
        print flat_action_index

        action = None
        idx = 0
        for empt in empty:
            if empt:
                if flat_action_index == idx:
                    action = flat_action_index
                    break
                else:
                    idx += 1
            elif not empt:
                idx += 1
                flat_action_index += 1

        print action
        ret = (action/BOARD_SIZE, action%BOARD_SIZE)
        print action, ret
        return ret

    def save_value_map(self):
        pkl.dump(self.values, open(self.filename, 'w'))

def ask_user():
    """Get coordinates from user via stdin return a tuple"""
    coords = sys.stdin.readline().rstrip("\n")
    x, y = coords.split(",")
    x = int(x)
    y = int(y)
    assert x < BOARD_SIZE
    assert y < BOARD_SIZE
    return (x, y)

def main():
    filename = None
    try:
        filename = sys.argv[1]
    except IndexError:
        pass
    agt = Agent(filename)
    brd = Board()
    while 1:
        while not brd.is_done():
            brd.set_cross(ask_user())
            if brd.is_done():
                agt.update_state(np.zeros((BOARD_SIZE, BOARD_SIZE)))
                break
            agt.set_state(brd.get_state_string())
            next_action = agt.get_action(brd.get_empty_fields())
            brd.set_circl(next_action)
            if brd.is_done():
                agt.update_state(np.ones((BOARD_SIZE, BOARD_SIZE)))
                break
            brd.print_board()
        brd.print_board()
        print brd.who_won()
        brd.clear()
        agt.save_value_map()

if __name__ == '__main__':
    main()

