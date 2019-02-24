from time import sleep
from math import inf
from random import randint

class ultimateTicTacToe:
    def __init__(self):
        """
        Initialization of the game.
        """
        self.board=[['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_']]
        self.maxPlayer='X'
        self.minPlayer='O'
        self.maxDepth=3
        #The start indexes of each local board
        self.globalIdx=[(0,0),(0,3),(0,6),(3,0),(3,3),(3,6),(6,0),(6,3),(6,6)]

        #Start local board index for reflex agent playing
        self.startBoardIdx=4
        #self.startBoardIdx=randint(0,8)

        #utility value for reflex offensive and reflex defensive agents
        self.winnerMaxUtility=10000
        self.twoInARowMaxUtility=500
        self.preventThreeInARowMaxUtility=100
        self.cornerMaxUtility=30

        self.winnerMinUtility=-10000
        self.twoInARowMinUtility=-100
        self.preventThreeInARowMinUtility=-500
        self.cornerMinUtility=-30

        self.expandedNodes=0
        self.currPlayer=True

        # Helper varaibles
        # All the possible winning sequences
        self.winning_sequences = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]

    def printGameBoard(self):
        """
        This function prints the current game board.
        """
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[:3]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[3:6]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[6:9]])+'\n')

    def local_board(self, currBoardIdx):
        """
        This function returns the current local board given its global board index.
        input args:
        currBoardIdx(int): int varaible indicates the index of the current board on the global board.
        output:
        board: the flatten local board for evaluation
        """
        row, col = self.globalIdx[currBoardIdx]
        board = []
        for i in range(row):
            for j in range(col):
                board.append(uttt.board[row + i][col + j])
        return board

    def evaluatePredifined(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for predifined agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        # Set up utility scores for different players
        if isMax:
            winner = self.winnerMaxUtility
            two_in_a_row = self.twoInARowMaxUtility
            prevent_three_in_a_row = self.preventThreeInARowMaxUtility
            corner = self.cornerMaxUtility
            player = self.maxPlayer
            opponent = self.minPlayer
        else:
            winner = self.winnerMinUtility
            two_in_a_row = self.twoInARowMinUtility
            prevent_three_in_a_row = self.preventThreeInARowMinUtility
            corner = self.cornerMinUtility
            player = self.minPlayer
            opponent = self.maxPlayer
        score=0

        # Loop through each local board
        for currBoardIdx in range(9):
            curr_board = local_board(self, currBoardIdx)
            for i, j, k in self.winning_sequences:
                    # First Rule: check if the player wins.
                    if curr_board[i] == curr_board[j] == curr_board[k] == player:
                        return winner
                    # Second Rule: check for unblocked two-in-a-rows and prevention of three in a row by the opponent
                    if curr_board[i] == curr_board[j] == player:
                        if curr_board[k] == opponent:
                            score += prevent_three_in_a_row
                        else:
                            score += two_in_a_row
                    if curr_board[i] == curr_board[k] == player:
                        if curr_board[j] == opponent:
                            score += prevent_three_in_a_row
                        else:
                            score += two_in_a_row
                    if curr_board[j] == curr_board[k] == player:
                        if curr_board[i] == opponent:
                            score += prevent_three_in_a_row
                        else:
                            score += two_in_a_row
            if score > 0:
                return score
            # Third Rule: check for corners
            if curr_board[0] == player:
                score += 30
        return score

    def make_move(self, row, col, currBoardIdx, isMax):
        """
        This function allow the player to make a move on the board.
        input args:
        row(int): int varaible indicates the row index of the local board in which the move is made.
        col(int): int varaible indicates the column index of the local board in which the move is made.
        currBoardIdx(int): int varaible indicates the index of the current board on the global board.
        isMax(boolean)
        output:
        (boolean): True if make move is successful
        """
        start_row, start_col = globalIdx[currBoardIdx]
        if self.board[start_row + row][start_col + col] != '_':
            return False
        if isMax:
            self.board[start_row + row][start_col + col] = self.maxPlayer
        else:
            self.board[start_row + row][start_col + col] = self.minPlayer
        return True

    def evaluateDesigned(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for your own agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        # Set up utility scores for different players
        if isMax:
            winner = self.winnerMaxUtility
            two_in_a_row = self.twoInARowMaxUtility
            prevent_three_in_a_row = self.preventThreeInARowMaxUtility
            corner = self.cornerMaxUtility
            player = self.maxPlayer
            opponent = self.minPlayer
        else:
            winner = self.winnerMinUtility
            two_in_a_row = self.twoInARowMinUtility
            prevent_three_in_a_row = self.preventThreeInARowMinUtility
            corner = self.cornerMinUtility
            player = self.minPlayer
            opponent = self.maxPlayer
        score=0

        # Loop through each local board
        for currBoardIdx in range(9):
            curr_board = local_board(self, currBoardIdx)
            for i, j, k in self.winning_sequences:
                    # First Rule: check if the player wins.
                    if curr_board[i] == curr_board[j] == curr_board[k] == player:
                        return winner
                    # Second Rule: check for unblocked two-in-a-rows and prevention of three in a row by the opponent
                    if curr_board[i] == curr_board[j] == player:
                        if curr_board[k] == opponent:
                            score += prevent_three_in_a_row
                        else:
                            score += two_in_a_row
                    if curr_board[i] == curr_board[k] == player:
                        if curr_board[j] == opponent:
                            score += prevent_three_in_a_row
                        else:
                            score += two_in_a_row
                    if curr_board[j] == curr_board[k] == player:
                        if curr_board[i] == opponent:
                            score += prevent_three_in_a_row
                        else:
                            score += two_in_a_row
            if score > 0:
                return score
            # Third Rule: check for corners
            if curr_board[0] == player:
                score += 30
        return score

    def checkMovesLeft(self):
        """
        This function checks whether any legal move remains on the board.
        output:
        movesLeft(bool): boolean variable indicates whether any legal move remains
                        on the board.
        """
        #YOUR CODE HERE
        movesLeft=False
        for row in self.board:
            for i in row:
                if i != self.maxPlayer and i != self.minPlayer:
                    return True
        return movesLeft

    def checkWinner(self):
        #Return termimnal node status for maximizer player 1-win,0-tie,-1-lose
        """
        This function checks whether there is a winner on the board.
        output:
        winner(int): Return 0 if there is no winner.
                     Return 1 if maxPlayer is the winner.
                     Return -1 if miniPlayer is the winner.
        """
        #YOUR CODE HERE
        winner=0
        # Check for max player first
        isMax = True
        if self.evaluatePredifined(True) == self.winnerMaxUtility:
            winner = 1
        elif self.evaluatePredifined(False) == self.winnerMinUtility:
            winner = -1
        return 0

    def alphabeta(self,depth,currBoardIdx,alpha,beta,isMax):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #YOUR CODE HERE
        if depth == self.maxDepth:
            return evaluatePredifined(currBoardIdx)
        # If either player wins, return best score
        if bestValue == self.winnerMaxUtility or bestValue == self.winnerMinUtility or not checkMovesLeft(self):
            return bestValue

        start_row, start_col = globalIdx[currBoardIdx]

        # If this is maxPlayer
        if isMax:
            bestValue = -inf
            # Traverse all cells
            for row in range(3):
                for col in range(3):
                    if self.board[row + start_row][col + start_col] == '_':
                        self.board[row + start_row][col + start_col] = self.maxPlayer
                        next_board_index = 3*row + col
                        isMax = not isMax
                        bestValue = max(bestValue, alphabeta(self, depth + 1, next_board_index, alpha, beta, isMax))
                        # Undo the move
                        if bestValue > beta:
                            self.board[row + start_row][col + start_col] == '_'
                            return bestValue
                        else:
                            alpha = bestValue
                        self.board[row + start_row][col + start_col] == '_'
            return bestValue
        else:
            bestValue = inf
            # Traverse all cells
            for row in range(3):
                for col in range(3):
                    if self.board[row + start_row][col + start_col] == '_':
                        self.board[row + start_row][col + start_col] = self.maxPlayer
                        next_board_index = 3*row + col
                        isMax = not isMax
                        bestValue = min(bestValue, alphabeta(self, depth + 1, next_board_index, alpha, beta, isMax))
                        if bestValue < alpha:
                            self.board[row + start_row][col + start_col] == '_'
                            return bestValue
                        else:
                            beta = bestValue
                        # Undo the move
                        self.board[row + start_row][col + start_col] == '_'
            return bestValue
        return bestValue

    def minimax(self, depth, currBoardIdx, isMax):
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #YOUR CODE HERE
        if depth == self.maxDepth:
            return evaluatePredifined(currBoardIdx)
        # If either player wins, return best score
        if bestValue == self.winnerMaxUtility or bestValue == self.winnerMinUtility or not checkMovesLeft(self):
            return bestValue

        start_row, start_col = globalIdx[currBoardIdx]

        # If this is maxPlayer
        if isMax:
            bestValue = -inf
            # Traverse all cells
            for row in range(3):
                for col in range(3):
                    if self.board[row + start_row][col + start_col] == '_':
                        self.board[row + start_row][col + start_col] = self.maxPlayer
                        next_board_index = 3*row + col
                        isMax = not isMax
                        bestValue = max(bestValue, minimax(self, depth + 1, next_board_index, isMax))
                        # Undo the move
                        self.board[row + start_row][col + start_col] == '_'
            return bestValue
        else:
            bestValue = inf
            # Traverse all cells
            for row in range(3):
                for col in range(3):
                    if self.board[row + start_row][col + start_col] == '_':
                        self.board[row + start_row][col + start_col] = self.maxPlayer
                        next_board_index = 3*row + col
                        isMax = not isMax
                        bestValue = min(bestValue, minimax(self, depth + 1, next_board_index, isMax))
                        # Undo the move
                        self.board[row + start_row][col + start_col] == '_'
            return bestValue
        return bestValue

    def find_best_move(self, currBoardIdx, isMax):
        start_row, start_col = globalIdx[currBoardIdx]
        if isMax:
            player = self.maxPlayer
            opponent = self.minPlayer
            bestValue = -inf
            for row in range(3):
                for col in range(3):
                    if self.board[row + start_row][col + start_col] == '_':
                        self.board[row + start_row][col + start_col] == player
                        move_value = minimax(self, 1, currBoardIdx isMax)
                        self.board[row + start_row][col + start_col] == '_'
                        if move_value > bestValue:
                            best_move = (row, col)
                            bestValue = move_value
        else:
            player = self.minPlayer
            opponent = self.maxPlayer
            bestValue = inf
            for row in range(3):
                for col in range(3):
                    if self.board[row + start_row][col + start_col] == '_':
                        self.board[row + start_row][col + start_col] == player
                        move_value = minimax(self, 1, currBoardIdx, isMax)
                        self.board[row + start_row][col + start_col] == '_'
                        if move_value < bestValue:
                            best_move = (row, col)
                            bestValue = move_value
        return best_move

    def playGamePredifinedAgent(self,maxFirst,isMinimaxOffensive,isMinimaxDefensive):
        """
        This function implements the processes of the game of predifined offensive agent vs defensive agent.
        input args:
        maxFirst(bool): boolean variable indicates whether maxPlayer or minPlayer plays first.
                        True for maxPlayer plays first, and False for minPlayer plays first.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for offensive agent.
                        True is minimax and False is alpha-beta.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for defensive agent.
                        True is minimax and False is alpha-beta.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        bestValue(list of float): list of bestValue at each move
        expandedNodes(list of int): list of expanded nodes at each move
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        bestValue=[]
        gameBoards=[]
        winner=0
        expandedNodes = 0

        currBoardIdx = self.startBoardIdx

        return gameBoards, bestMove, expandedNodes, bestValue, winner

    def playGameYourAgent(self):
        """
        This function implements the processes of the game of your own agent vs predifined offensive agent.
        input args:
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        gameBoards=[]
        winner=0
        return gameBoards, bestMove, winner


    def playGameHuman(self):
        """
        This function implements the processes of the game of your own agent vs a human.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        gameBoards=[]
        winner=0
        return gameBoards, bestMove, winner

if __name__=="__main__":
    uttt=ultimateTicTacToe()
    gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(True,False,False)
    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")
