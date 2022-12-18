import sys
n = int(input("Enter the value of n : "))
if n <= 3:
    print("please enter the size greater than 3")
    sys.exit()
board = []


def getBoard():
    for i in range(n):
        List = []
        for j in range(n):
            List.append(0)
        board.append(List)


def printBoard():
    for i in board:
        print(str(i).replace(',', '').replace('\'', ''))
    print('------------------------------------------------------------------')


def isSafe(row, col):
    for i in range(n):
        if board[row][i] == 1:
            return False
    for j in range(n):
        if board[j][col] == 1:
            return False


# check left up diagonal
    i = row-1
    j = col-1
    while(i >= 0 and j >= 0):
        if board[i][j] == 1:
            return False
        i = i-1
        j = j-1
# check right up diagonal

    i = row-1
    j = col+1
    while(i >= 0 and j < n):
        if board[i][j] == 1:
            return False
        i = i-1
        j = j+1

    i = row+1
    j = col-1
    while(i < n and j >= 0):
        if board[i][j] == 1:
            return False
        i = i+1
        j = j-1

    i = row+1
    j = col+1
    while(i < n and j < n):
        if board[i][j] == 1:
            return False
        i = i+1
        j = j+1
    return True


def Put(n, count):
    if count == n:
        printBoard()
        return

    for i in range(len(board)):
        if isSafe(count, i):
            board[count][i] = 1
            Put(n, count + 1)
            board[count][i] = 0


getBoard()
Put(n, 0)
