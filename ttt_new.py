import copy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
BOARD_SIZE = 3
EXP_STEPS = 100
ALPHA = 0.5
BLOCKING_EXPERT = False
RUNS = 10000
DO_PRINT = False
EPISODE_LENGTH = 500


def number_of_matches(match):
    return len(match[0])


def map_pseudo_to_normal(pseudo_row, pseudo_col):
    """Some functions use a pseud-row coordinate system. This system consists
    of rows and columns but the columns are defined in the sense of the
    function extract_all_rows. Columns are then the index within that row. This
    function maps those coordinates back to the normal numpy ones."""
    # the first BS pseudo rows are actual rows
    if pseudo_row < BOARD_SIZE:
        return pseudo_row, pseudo_col
    # the second set of BS rows are actual columns
    elif BOARD_SIZE <= pseudo_row < 2 * BOARD_SIZE:
        # here, we just need to transpose them
        return pseudo_col, (pseudo_row - BOARD_SIZE)
    # the last to indices stand for diagonal and anti-diagonal, respectively
    elif pseudo_row == 2*BOARD_SIZE:
        # the default diagonal
        idx = 1
        return idx * pseudo_col, idx * pseudo_col
    elif pseudo_row == 2*BOARD_SIZE+1:
        # the anti-diagonal, x increases, y decreases
        idx = 0
        idy = BOARD_SIZE - 1
        return idy - pseudo_col, idx + pseudo_col


def extract_all_rows(board):
    """Generator function for all rows, columns and diagonals of the board"""
    # return the rows first
    for row in range(BOARD_SIZE):
        yield board[row, :]
    # then the columns
    for col in range(BOARD_SIZE):
        yield board[:, col]
    # then the diagonal
    yield np.diagonal(board)
    # and last the anti-diagonal
    yield np.diagonal(np.flipud(board))


def to_string(a):
    return ", ".join([str(value) for value in a.flatten()])


def experienced_player(my_mark, op_mark, board):
    """Takes two marks (symbol value) for self and opponent and a board and returns
    a board with an additional mark."""
    match = np.where(board > 0)
    # This if statement implements a case on match
    if number_of_matches(match) == 0:
        # empty board, choose center
        board[1, 1] = my_mark
        return board
    elif number_of_matches(match) == 1:
        # first move taken as center:
        if board[1, 1] == 0:
            board[1, 1] = my_mark
            return board
        else:
            for (x, y) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                if board[y, x] == 0:
                    board[y, x] = my_mark
                    return board
    else:
        # if there are two marks or more, iterate over all rows, cols, diagonals
        # store rows that can be completed and rows that need to be blocked
        completable = []
        blockable = []
        for (num, values) in enumerate(extract_all_rows(board)):
            # if the row is complete, take the next one
            number_of_marks = number_of_matches(np.where(values > 0))
            if number_of_marks == 3:
                continue
            # if the row can be completed by this player, then do so
            my_marks = np.where(values == my_mark)
            # print my_marks
            if number_of_matches(my_marks) == 2:
                completable.append((num, my_marks))
            op_marks = np.where(values == op_mark)
            # print op_marks
            if number_of_matches(op_marks) == 2:
                blockable.append((num, op_marks))

        verbs = ["completing"]
        lists = [completable]
        if BLOCKING_EXPERT:
            verbs.append("blocking")
            lists.append(blockable)

        for verb, lst in zip(verbs, lists):
            if len(lst) > 0:
                free_space = lst[0]
                # print free_space
                for col in range(BOARD_SIZE):
                    if col in free_space[1][0]:
                        continue
                    else:
                        # print verb,"field",free_space[0], col
                        (y, x) = map_pseudo_to_normal(free_space[0], col)
                        board[y, x] = my_mark
                        return board

        # last resort: randomness
        # print "Random"
        empty_fields = np.where(board == 0)
        field = np.random.randint(len(empty_fields[0]))
        board[empty_fields[0][field], empty_fields[1][field]] = my_mark
        return board


def is_win(values):
    """Check whether a 1d-array contains equal elements greater than zero"""
    if values[0] == 0:
        return 0
    for idx in range(1, BOARD_SIZE):
        if not values[idx] == values[idx - 1]:
            return 0
    return values[0]


def win_or_tie(board):
    """Check if the board was won (and return the winner) or if it was a tie
    (and return -1) of if it's still open (and return None)"""
    # Nice and compact through the use of generators
    for values in extract_all_rows(board):
        res = is_win(values)
        if res > 0:
            return res
    # check if it's a tie, else return None
    if number_of_matches(np.where(board > 0)) == 9:
        return -1
    else:
        return None


def get_value(vm, my_mark, op_mark, board):
    try:
        return vm[to_string(board)]
    except KeyError:
        res = win_or_tie(board)
        if res == my_mark:
            return 1.0
        elif res == op_mark:
            return 0.0
        else:
            return 0.5


def update_value_map(vm, new_value, last_state):
    key = to_string(last_state)
    if key not in vm:
        vm[key] = 0.5

    vm[key] = vm[key] + ALPHA*(new_value-vm[key])

    return vm


def reinforcement_player(my_mark, op_mark, board, last_state, vm):
    # Find out the empty positions
    empty_positions = np.where(board == 0)
    if DO_PRINT:
        print("Empty board positions: {}".format(empty_positions))
    # Decide whether to do an exploratory or a greedy move
    if np.random.randint(EXP_STEPS) == 0:
        # Explore
        selected = np.random.randint(len(empty_positions))
        if DO_PRINT:
            print("Random move: ", empty_positions[0][selected], empty_positions[1][selected])
        board[empty_positions[0][selected], empty_positions[1][selected]] = my_mark
        return board, vm
    else:
        # Follow the value
        values = []
        # Calculate the value of each configuration resulting from a move on
        # any empty field
        for y, x in zip(empty_positions[0], empty_positions[1]):
            new_board = copy.deepcopy(board)
            new_board[y, x] = my_mark
            value = get_value(vm, my_mark, op_mark, new_board)
            # if the resulting value is already 1.0 then just take it
            if value == 1.0:
                # done
                vm = update_value_map(vm, value, last_state)
                return new_board, vm
            else:
                # otherwise collect the value for later
                values.append(value)
        # select maximum value, however if all values are equal then pick a
        # random field
        values = np.array(values)
        match0 = np.where(values == values[0])
        if len(match0[0]) == len(values):
            pick = np.random.randint(len(values))
            board[empty_positions[0][pick], empty_positions[1][pick]] = my_mark
            vm = update_value_map(vm, values[pick], last_state)
        else:
            max_idx = np.argmax(values)
            board[empty_positions[0][max_idx], empty_positions[1][max_idx]] = my_mark
            vm = update_value_map(vm, values[max_idx], last_state)
        return board, vm


def main():
    global DO_PRINT
    # Initialize the value map
    value_map = {}
    winning = []
    print()
    for run in range(RUNS):
        # if run >= RUNS-5:
        #     DO_PRINT = True
        if run % 100 == 0 or DO_PRINT:
            print("Run", run)
        # Initialize a board
        board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        result = None
        last_state = copy.deepcopy(board)
        while result is None:
            if DO_PRINT:
                print(board)
                # Let the experienced bot move first
                print("Player 1 (bot)")
            board = experienced_player(1, 2, board)
            # Check if it was won
            result = win_or_tie(board)
            if result is not None:
                if result > 0:
                    if DO_PRINT:
                        print("Player", result, "won")
                    winning.append(1)
                else:
                    if DO_PRINT:
                        print("Tie")
                    winning.append(0)
                break
            # let the reinforcement player move second
            if DO_PRINT:
                print(board)
                print("Player 2 (learner)")
            board, value_map = reinforcement_player(2, 1, board, last_state, value_map)
            last_state = copy.deepcopy(board)
            result = win_or_tie(board)
            if result is not None:
                if result > 0:
                    if DO_PRINT:
                        print("Player", result, "won")
                    winning.append(2)
                else:
                    if DO_PRINT:
                        print("Tie")
                    winning.append(0)
                break
        if DO_PRINT:
            print("Board at the end of the game:")
            print(board)

    print()
    print("---------------------------")
    print(" End of training reporting")
    print("---------------------------")
    print(" Value map had {} entries.".format(len(value_map)))
    print(" Entries are:")
    for key, val in value_map.items():
        print("{}: {}".format(key, val))

    print(" Overall results were:")
    results = np.bincount(winning, minlength=3)
    print("  Draw    : {:5d}, {:5.2f}".format(results[0], results[0] / RUNS * 100.0))
    print("  Player 1: {:5d}, {:5.2f}".format(results[1], results[1] / RUNS * 100.0))
    print("  Player 2: {:5d}, {:5.2f}".format(results[2], results[2] / RUNS * 100.0))

    num_episodes = int(RUNS/EPISODE_LENGTH)
    fractions = np.zeros((num_episodes, 3))
    for episode in range(num_episodes):
        results_of_episode = winning[episode * EPISODE_LENGTH:(episode + 1) * EPISODE_LENGTH]
        fractions[episode] = np.array(np.bincount(results_of_episode, minlength=3), dtype='float')/EPISODE_LENGTH

    if not DO_PRINT:
        plt.plot(fractions[:, 0], label='draw')
        plt.plot(fractions[:, 1], label='Player 1 (exp.)')
        plt.plot(fractions[:, 2], label='Player 2 (learn)')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
