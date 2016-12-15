import numpy as np

BS = 3

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
    elif pseudo_row > BS and pseudo_row < 2*BS:
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
                if board[x,y] == 0:
                    board[x,y] = my_mark
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

        for lst in [completable, blockable]:
            if len(lst) > 0:
                free_space = lst[0]
                print free_space
                for col in range(BS):
                    if col in free_space[1][0]:
                        continue
                    else:
                        print free_space[0], col
                        (x,y) = map_pseudo_to_normal(free_space[0], col)
                        board[x,y] = my_mark
                        return board

        # last resort: randomness
        
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
    
def main():
    # Initialize a board
    board = np.zeros((BS,BS))
    board = np.random.randint(3, size=(BS,BS))
    print()
    print(board)
    # Check if it was won
    result = win_or_tie(board)
    if result is not None:
        if result > 0:
            print "Player",result,"won"
        else:
            print "Tie"
        return result
    # Let the bot move
    board = experienced_player(1,2,board)
    print board

if __name__ == '__main__':
    main()

main()

