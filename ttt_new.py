import copy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
BS = 3
EXP_STEPS = 5
ALPHA = 0.5
BLOCKING_EXPERT = False
RUNS = 20000
EPISODE_LENGTH = 1000

def nummatch(match):
    return len(match[0])

def map_pseudo_to_normal(pseudo_row, pseudo_col):
    '''Some functions use a pseud-row coordinate system. This system consists
    of rows and columns but the columns are defined in the sense of the
    function extract_all_rows. Columns are then the index within that row. This
    function maps those coordinates back to the normal numpy ones.'''
    # the first BS pseudo rows are actual rows
    if pseudo_row < BS:
        return (pseudo_row, pseudo_col)
    # the second set of BS rows are actual columns
    elif pseudo_row >= BS and pseudo_row < 2*BS:
        # here, we just need to transpose them
        return (pseudo_col, pseudo_row-BS)
    # the last to indices stand for diagonal and anti-diagonal, respectively
    elif pseudo_row == 2*BS:
        # the default diagonal
        idx = 1
        return (idx*pseudo_col, idx*pseudo_col)
    elif pseudo_row == 2*BS+1:
        # the anti-diagonal, x increases, y decreases
        idx = 0
        idy = BS-1
        return (idy-pseudo_col, idx+pseudo_col)
    
def extract_all_rows(board):
    '''Generator function for all rows, columns and diagonals of the board'''
    # return the rows first
    for row in range(BS):
        yield board[row,:]
    # then the columns
    for col in range(BS):
        yield board[:,col]
    # then the diagonal
    yield np.diagonal(board)
    # and last the anti-diagonal
    yield np.diagonal(np.flipud(board))
    
def experienced_player(my_mark, op_mark, board):
    '''Takes two marks (symbol value) for self and opponent and a board and returns
    a board with an additional mark.'''
    match = np.where(board>0)
    # This if statement implements a case on match
    if nummatch(match) == 0:
        # empty board, choose center
        board[1,1] = my_mark
        return board
    elif nummatch(match) == 1:
        # first move taken as center:
        if board[1,1] == 0:
            board[1,1] = my_mark
            return board
        else:
            for (x,y) in [(0,0),(0,2),(2,0),(2,2)]:
                if board[y,x] == 0:
                    board[y,x] = my_mark
                    return board
    else:
        # if there are two marks or more, iterate over all rows, cols, diags
        # store rows that can be completed and rows that need to be blocked
        completable = []
        blockable = []
        for (num,vals) in enumerate(extract_all_rows(board)):
            # if the row is complete, take the next one
            nummarks = nummatch(np.where(vals > 0))
            if nummarks == 3:
                continue
            # if the row can be completed by this player, then do so
            mymarks = np.where(vals == my_mark)
            #print mymarks
            if nummatch(mymarks) == 2:
                completable.append((num,mymarks))
            opmarks = np.where(vals == op_mark)
            #print opmarks
            if nummatch(opmarks) == 2:
                blockable.append((num,opmarks))

        verbs = ["completing"]
        lists = [completable]
        if BLOCKING_EXPERT:
            verbs.append("blocking")
            lists.append(blockable)

        for verb,lst in zip(verbs,lists):
            if len(lst) > 0:
                free_space = lst[0]
                #print free_space
                for col in range(BS):
                    if col in free_space[1][0]:
                        continue
                    else:
                        #print verb,"field",free_space[0], col
                        (y,x) = map_pseudo_to_normal(free_space[0], col)
                        board[y,x] = my_mark
                        return board

        # last resort: randomness
        #print "Random"
        empty_fields = np.where(board == 0)
        field = np.random.randint(len(empty_fields[0]))
        board[empty_fields[0][field],empty_fields[1][field]] = my_mark
        return board

def is_win(vals):
    '''Check whether a 1darray contains equal elements greater than zero'''
    if vals[0] == 0:
        return 0
    for idx in range(1,BS):
        if not vals[idx] == vals[idx-1]:
            return 0
    return vals[0]

def win_or_tie(board):
    '''Check if the board was won (and return the winner) or if it was a tie
    (and return -1) of if it's still open (and return None)'''
    # Nice and compact through the use of generators
    for vals in extract_all_rows(board):
        res = is_win(vals)
        if res > 0:
            return res
    # check if it's a tie, else return None
    if nummatch(np.where(board>0)) == 9:
        return -1
    else:
        return None


def get_value(vm, my_mark, op_mark, board):
    try:
        return vm[board.tostring()]
    except KeyError:
        res = win_or_tie(board)
        if res == my_mark:
            return 1.0
        elif res == op_mark:
            return 0.0
        else:
            return 0.5

def update_value_map(vm, new_value, last_state):
    key = last_state.tostring()
    if key in vm:
        vm[key] = vm[key] + ALPHA*(new_value-vm[key])
    else:
        vm[key] = 0.5 + ALPHA*(new_value-0.5)
    return vm

def reinforcement_player(my_mark, op_mark, board, last_state, vm):
    # Find out the empty positions
    empty = np.where(board==0)
    # Decide whether to move exploratorily or greedily
    if np.random.randint(EXP_STEPS) == 0:
        # Explore
        selected = np.random.randint(len(empty))
        #print "Random",empty[0][selected],empty[1][selected]
        board[empty[0][selected],empty[1][selected]] = my_mark
        return board, vm
    else:
        # Follow the value
        values = []
        # Calculate the value of each configuration resulting from a move on
        # any empty field
        for y,x in zip(empty[0],empty[1]):
            new_board = copy.copy(board)
            new_board[y,x] = my_mark
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
        match0 = np.where(values==values[0])
        if len(match0[0]) == len(values):
            pick = np.random.randint(len(values))
            board[empty[0][pick],empty[1][pick]] = my_mark
            vm = update_value_map(vm, values[pick], last_state)
        else:
            max_idx = np.argmax(values)
            board[empty[0][max_idx],empty[1][max_idx]] = my_mark
            vm = update_value_map(vm, values[max_idx], last_state)
        return board, vm
    
def main():
    # Initialize the value map
    value_map = {}
    winning = []
    print()
    for run in range(RUNS):
        if run%100 == 0:
            print "Run",run
        # Initialize a board
        board = np.zeros((BS,BS))
        last_state = board
        result = None
        while result is None:
            #print(board)
            # Let the experienced bot move first
            #print "Player 1 (bot)"
            board = experienced_player(1,2,board)
            # Check if it was won
            result = win_or_tie(board)
            if result is not None:
                if result > 0:
                    #print "Player",result,"won"
                    winning.append(1)
                else:
                    #print "Tie"
                    winning.append(0)
                break
            # let the reinforcement player move second
            #print "Player 2 (learner)"
            board, value_map = reinforcement_player(2,1,board,last_state,value_map)
            last_state = copy.copy(board)
            result = win_or_tie(board)
            if result is not None:
                if result > 0:
                    #print "Player",result,"won"
                    winning.append(2)
                else:
                    #print "Tie"
                    winning.append(0)
                break
        #print board

    for key,val in value_map.iteritems():
        print np.fromstring(key), val

    print np.bincount(winning,minlength=3)
    fractions = np.zeros((RUNS/EPISODE_LENGTH,3))
    for episode in range(RUNS/EPISODE_LENGTH):
        fractions[episode] = np.array(np.bincount(winning[episode*EPISODE_LENGTH:(episode+1)*EPISODE_LENGTH],
                                                  minlength=3), dtype='float')/EPISODE_LENGTH
    plt.plot(fractions[:,0], label='draw')
    plt.plot(fractions[:,1], label='Player 1 (exp.)')
    plt.plot(fractions[:,2], label='Player 2 (learn)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()


