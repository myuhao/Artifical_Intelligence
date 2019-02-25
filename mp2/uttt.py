from time import sleep
from math import inf
from random import randint
import copy # copy.deepcopy()

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
        self.global_board = ['_','_','_','_','_','_','_','_','_']

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
        for i in range(3):
            for j in range(3):
                board.append(self.board[row + i][col + j])
                # print(board)
        # print("Reached line 69")
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


        for currBoardIdx in range(9):
            curr_board = self.local_board(currBoardIdx)
            num_two_in_a_row = 0
            num_prevent_three_in_a_row = 0
            # print(curr_board)
            # pdb.set_trace()
            for i, j, k in self.winning_sequences:
                # First Rule: check if the player wins.
                if curr_board[i] == curr_board[j] == curr_board[k] == player:
                    return winner
                # Second Rule: check for unblocked two-in-a-rows
                if curr_board[i] == curr_board[j] == player:
                    if curr_board[k] == '_':
                        num_two_in_a_row += 1
                if curr_board[i] == curr_board[k] == player:
                    if curr_board[j] == '_':
                        num_two_in_a_row += 1
                if curr_board[j] == curr_board[k] == player:
                    if curr_board[i] == '_':
                        num_two_in_a_row += 1
                # and prevention of three in a row by the opponent
                if curr_board[i] == curr_board[j] == opponent and curr_board[k] == player:
                        num_prevent_three_in_a_row += 1
                if curr_board[i] == curr_board[k] == opponent and curr_board[j] == player:
                        num_prevent_three_in_a_row += 1
                if curr_board[j] == curr_board[k] == opponent and curr_board[i] == player:
                        num_prevent_three_in_a_row += 1
            if  num_two_in_a_row > 0 or num_prevent_three_in_a_row > 0:
                score += num_two_in_a_row*two_in_a_row + num_prevent_three_in_a_row*prevent_three_in_a_row
            else:
                corners = [0, 2, 6, 8]
                for corner in corners:
                    if curr_board[corner] == player:
                        score += 30
            # return score
        # Third Rule: check for corners
        # for currBoardIdx in range(9):
        #     curr_board = self.local_board(currBoardIdx)
        #     corners = [0, 2, 6, 8]
        #     for i in corners:
        #         if curr_board[i] == player:
        #             score += 30
        return score


        # Loop through each local board
        # for currBoardIdx in range(9):
        #     curr_board = self.local_board(currBoardIdx)
        #     # print(curr_board)
        #     # pdb.set_trace()
        #     for i, j, k in self.winning_sequences:
        #             # First Rule: check if the player wins.
        #             if curr_board[i] == curr_board[j] == curr_board[k] == player:
        #                 return winner
        #             # Second Rule: check for unblocked two-in-a-rows
        #             if curr_board[i] == curr_board[j] == player:
        #                 if curr_board[k] == '_':
        #                     score += two_in_a_row
        #             if curr_board[i] == curr_board[k] == player:
        #                 if curr_board[j] == '_':
        #                     score += two_in_a_row
        #             if curr_board[j] == curr_board[k] == player:
        #                 if curr_board[i] == '_':
        #                     score += two_in_a_row
        #             # and prevention of three in a row by the opponent
        #             if curr_board[i] == curr_board[j] == opponent:
        #                 if curr_board[k] == player:
        #                     score += two_in_a_row
        #             if curr_board[i] == curr_board[k] == opponent:
        #                 if curr_board[j] == player:
        #                     score += two_in_a_row
        #             if curr_board[j] == curr_board[k] == opponent:
        #                 if curr_board[i] == player:
        #                     score += two_in_a_row
        #     if score > 0:
        #         return score
        #     # Third Rule: check for corners
        #     corners = [0, 2, 6, 8]
        #     for i in corners:
        #         if curr_board[i] == player:
        #             score += 30
        # return score

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
        start_row, start_col = self.globalIdx[currBoardIdx]
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
        player = self.minPlayer

        opponent_winner = self.winnerMaxUtility
        opponent_two_in_a_row = self.twoInARowMaxUtility
        opponent_prevent_three_in_a_row = self.preventThreeInARowMaxUtility
        opponent_corner = self.cornerMaxUtility
        opponent_player = self.maxPlayer

        score=0
        num_two_in_a_row = 0
        num_prevent_three_in_a_row = 0
        num_corners = 0

        for currBoardIdx in range(9):
            curr_board = self.local_board(currBoardIdx)
            # print(curr_board)
            # pdb.set_trace()
            for i, j, k in self.winning_sequences:
                    # First Rule: check if the player wins.
                    if curr_board[i] == curr_board[j] == curr_board[k] == player:
                        return 10000

                    if curr_board[i] == curr_board[j] == curr_board[k] == opponent_player:
                        return -10000

                    # Second Rule: check for unblocked two-in-a-rows
                    if curr_board[i] == curr_board[j] == player:
                        if curr_board[k] == '_':
                            num_two_in_a_row += 1
                    if curr_board[i] == curr_board[k] == player:
                        if curr_board[j] == '_':
                            num_two_in_a_row += 1
                    if curr_board[j] == curr_board[k] == player:
                        if curr_board[i] == '_':
                            num_two_in_a_row += 1

                    if curr_board[i] == curr_board[j] == opponent_player:
                        if curr_board[k] == '_':
                            num_two_in_a_row -= 1.2
                    if curr_board[i] == curr_board[k] == opponent_player:
                        if curr_board[j] == '_':
                            num_two_in_a_row -= 1.2
                    if curr_board[j] == curr_board[k] == opponent_player:
                        if curr_board[i] == '_':
                            num_two_in_a_row -= 1.2

                    # and prevention of three in a row by the opponent
                    if curr_board[i] == curr_board[j] == opponent_player:
                        if curr_board[k] == player:
                            num_prevent_three_in_a_row -= 1
                    if curr_board[i] == curr_board[k] == opponent_player:
                        if curr_board[j] == player:
                            num_prevent_three_in_a_row -= 1
                    if curr_board[j] == curr_board[k] == opponent_player:
                        if curr_board[i] == player:
                            num_prevent_three_in_a_row -= 1
                    score -= num_two_in_a_row*500 - num_prevent_three_in_a_row*100
                    corners = [0, 2, 6, 8]
                    if curr_board[i] == player:
                        for i in corners:
                                score += 30
        # if  num_two_in_a_row > 0 or num_prevent_three_in_a_row > 0:
        #     score += num_two_in_a_row*500 + num_prevent_three_in_a_row*100
        #     return score
        # # Third Rule: check for corners
        # for currBoardIdx in range(9):
        #     curr_board = self.local_board(currBoardIdx)
        #     corners = [0, 2, 6, 8]
        #     for i in corners:
        #         if curr_board[i] == player:
        #             score += 30

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
        if self.evaluatePredifined(True) >= self.winnerMaxUtility:
            winner = 1
        elif self.evaluatePredifined(False) <= self.winnerMinUtility:
            winner = -1
        return winner

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
            return self.evaluatePredifined(currBoardIdx)
        # If either player wins, return best score

        start_row, start_col = self.globalIdx[currBoardIdx]

        # If this is maxPlayer
        if isMax:
            bestValue = -inf
            # Traverse all cells
            for row in range(3):
                for col in range(3):
                    if self.board[row + start_row][col + start_col] == '_':
                        self.board[row + start_row][col + start_col] = self.maxPlayer
                        if self.evaluatePredifined(currBoardIdx) == self.winnerMaxUtility or self.evaluatePredifined(currBoardIdx) == self.winnerMinUtility or not self.checkMovesLeft():
                            self.board[row + start_row][col + start_col] = '_'
                            return self.evaluatePredifined(currBoardIdx)
                        next_board_index = 3*row + col
                        isMax = not isMax
                        bestValue = max(bestValue, self.alphabeta(depth + 1, next_board_index, alpha, beta, isMax))
                        # Undo the move
                        if bestValue > beta:
                            self.board[row + start_row][col + start_col] = '_'
                            return bestValue
                        else:
                            alpha = bestValue
                        self.board[row + start_row][col + start_col] = '_'
            return bestValue
        else:
            bestValue = inf
            # Traverse all cells
            for row in range(3):
                for col in range(3):
                    if self.board[row + start_row][col + start_col] == '_':
                        self.board[row + start_row][col + start_col] = self.minPlayer
                        if self.evaluatePredifined(currBoardIdx) == self.winnerMaxUtility or self.evaluatePredifined(currBoardIdx) == self.winnerMinUtility or not self.checkMovesLeft():
                            self.board[row + start_row][col + start_col] = '_'
                            return self.evaluatePredifined(currBoardIdx)
                        next_board_index = 3*row + col
                        isMax = not isMax
                        bestValue = min(bestValue, self.alphabeta(depth + 1, next_board_index, alpha, beta, isMax))
                        if bestValue < alpha:
                            self.board[row + start_row][col + start_col] = '_'
                            return bestValue
                        else:
                            beta = bestValue
                        # Undo the move
                        self.board[row + start_row][col + start_col] = '_'
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
            return self.evaluatePredifined(currBoardIdx)


        start_row, start_col = self.globalIdx[currBoardIdx]

        # If this is maxPlayer
        if isMax:
            bestValue = -inf
            # Traverse all cells
            for row in range(3):
                for col in range(3):
                    if self.board[row + start_row][col + start_col] == '_':
                        self.board[row + start_row][col + start_col] = self.maxPlayer
                        # If either player wins, return best score
                        if self.evaluatePredifined(currBoardIdx) == self.winnerMaxUtility or self.evaluatePredifined(currBoardIdx) == self.winnerMinUtility or not self.checkMovesLeft():
                            self.board[row + start_row][col + start_col] = '_'
                            return self.evaluatePredifined(currBoardIdx)
                       # pdb.set_trace()
                        next_board_index = 3*row + col
                        curr_value = self.minimax(depth + 1, next_board_index, not isMax)
                        if curr_value > bestValue:
                            bestValue = curr_value
                        #self.board[row + start_row][col + start_col] == '_'
                      #  pdb.set_trace()
                        # Undo the move
                        self.board[row + start_row][col + start_col] = '_'
            return bestValue
        else:
            bestValue = inf
            # Traverse all cells
            for row in range(3):
                for col in range(3):
                    if self.board[row + start_row][col + start_col] == '_':
                        self.board[row + start_row][col + start_col] = self.minPlayer
                        # If either player wins, return best score
                        if self.evaluatePredifined(currBoardIdx) == self.winnerMaxUtility or self.evaluatePredifined(currBoardIdx) == self.winnerMinUtility or not self.checkMovesLeft():
                            self.board[row + start_row][col + start_col] = '_'
                            return self.evaluatePredifined(currBoardIdx)
                        next_board_index = 3*row + col
                        # bestValue = min(bestValue, self.minimax(depth + 1, next_board_index, not isMax))

                        # Undo the move
                        curr_value = self.minimax(depth + 1, next_board_index, not isMax)
                        if curr_value < bestValue:
                            bestValue = curr_value
                        self.board[row + start_row][col + start_col] = '_'
            return bestValue
        return bestValue

    def find_best_move(self, currBoardIdx, isMax, isMinimax):
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        currBoardIdx(int): current local board index
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        isMinimax(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm
        output:
        best_move(tuple):the coordinates of the best move
        """
        start_row, start_col = self.globalIdx[currBoardIdx]
        best_move = (0, 0)
        if isMax:
            player = self.maxPlayer
            opponent = self.minPlayer
            bestValue = -inf
            for row in range(3):
                for col in range(3):
                    if self.board[row + start_row][col + start_col] == '_':
                        self.board[row + start_row][col + start_col] = player
                        if isMinimax:
                            move_value = self.minimax(1, currBoardIdx, isMax)
                        else:
                            move_value = self.alphabeta(1, currBoardIdx, -inf, inf, isMax)
                        if move_value > bestValue:
                            best_move = (row, col)
                            bestValue = move_value
                        self.board[row + start_row][col + start_col] = '_'
        else:
            player = self.minPlayer
            opponent = self.maxPlayer
            bestValue = inf
            for row in range(3):
                for col in range(3):
                    if self.board[row + start_row][col + start_col] == '_':
                        self.board[row + start_row][col + start_col] = player
                        if isMinimax:
                            move_value = self.minimax(1, currBoardIdx, isMax)
                        else:
                            move_value = self.alphabeta(1, currBoardIdx, -inf, inf, isMax)
                        if move_value < bestValue:
                            best_move = (row, col)
                            bestValue = move_value
                        self.board[row + start_row][col + start_col] = '_'
        return best_move

    def _minimax(self, depth, currBoardIdx, isMax):
        bestValue = 0
        if depth == self.maxDepth:
            return self.evaluatePredifined(isMax)

        startRow, startCol = self.globalIdx[currBoardIdx]
        for localRow in range(3):
            for localCol in range(3):
                if self.board[startRow+localRow][startCol+localCol] == "_":
                    self.board[startRow+localRow][startCol+localCol] = self.maxPlayer if isMax else self.minPlayer
                    nextBoardIndex = localRow*3 + localCol
                    newValue = self._minimax(depth+1, nextBoardIndex, not isMax)
                    if newValue > bestValue:
                        bestValue = newValue
        return bestValue

    def _findBestMove(self, currBoardIdx, isMax, isMinimax):
        """
        This function use the algorithm to find the best move for the current player at the
        designated board.

        Arguments:
            currBoardIdx: int -- board
            isMax: bool -- Which player we are calculating. True for Max/"X", False for Min/"O"
            isMinimax: bool -- Are we using Minimax or alphaBeta.

        Returns:
            bestMove: tuple -- The local coord of the best move in the currBoardIdx.
        """
        bestMove = (0,0)
        bestValue = -inf
        startRow, startCol = self.globalIdx[currBoardIdx]
        for localRow in range(3):
            for localCol in range(3):
                if self.board[startRow+localRow][startCol+localCol] == "_":
                    self.board[startRow+localRow][startCol+localCol] = self.maxPlayer if isMax else self.minPlayer
                    newValue = self._minimax(1, currBoardIdx, isMax)
                    if newValue > bestValue:
                        bestValue = newValue
                        bestMove = (localRow, localCol)
                    self.board[startRow+localRow][startCol+localCol] = "_"
        return bestMove

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
        i = 0
        while self.checkMovesLeft():

            if i%1 == 0:
                self.printGameBoard()
                print(self.checkWinner())
            if maxFirst: #max first sequence
                best_move = self.find_best_move(currBoardIdx,True,isMinimaxOffensive)
                self.make_move(best_move[0], best_move[1],currBoardIdx,True)
                #update result arrays


                if self.checkWinner() != 0:
                    winner = self.checkWinner()
                    break
                currBoardIdx = 3*best_move[0]+best_move[1]

                best_move = self.find_best_move(currBoardIdx,False,isMinimaxDefensive)
                self.make_move(best_move[0],best_move[1],currBoardIdx,False)
                if self.checkWinner() != 0:
                    winner = self.checkWinner()
                    break
                currBoardIdx = 3*best_move[0]+best_move[1]

            else: #min first sequence
                best_move = self.find_best_move(currBoardIdx,False,isMinimaxDefensive)
                self.make_move(best_move[0],best_move[1],currBoardIdx,False)
                if self.checkWinner() !=0:
                    winner = self.checkWinner()
                    break
                currBoardIdx = 3*best_move[0], best_move[1]

                best_move = self.find_best_move(currBoardIdx, True, isMinimaxOffensive)
                self.make_move(best_move[0], best_move[1], currBoardIdx, True)
                if self.checkWinner() != 0:
                    winner = self.checkWinner()
                    break
                currBoardIdx = 3 * best_move[0] + best_move[1]
            i += 1

        return gameBoards, bestMove, expandedNodes, bestValue, winner

    def myPlayGamePredifinedAgent(self,maxFirst,isMinimaxOffensive,isMinimaxDefensive):
        bestMove=[]
        bestValue=[]
        gameBoards=[]
        winner=0
        expandedNodes = 0

        currBoardIdx = self.startBoardIdx
        isMax = maxFirst
        while self.checkMovesLeft() == True:
            move = self._findBestMove(currBoardIdx, isMax, isMinimaxOffensive) # move is a local coords
            startRow, startCol = self.globalIdx[currBoardIdx]
            localRow = move[0]
            localCol = move[1]
            self.board[startRow+localRow][startCol+localCol] = self.maxPlayer if isMax else self.minPlayer
            isMax = not isMax
            currBoardIdx = localRow*3 + localCol
            if self.checkWinner() != 0:
                winner = self.checkWinner()
                break
        return gameBoards, bestMove, expandedNodes, bestValue, winner

    def alphabeta_designed(self,depth,currBoardIdx,alpha,beta,isDesigned):
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
            return self.evaluateDesigned(currBoardIdx)
        # If either player wins, return best score

        start_row, start_col = self.globalIdx[currBoardIdx]

        # If this is maxPlayer
        if isDesigned:
            bestValue = -inf
            # Traverse all cells
            for row in range(3):
                for col in range(3):
                    if self.board[row + start_row][col + start_col] == '_':
                        self.board[row + start_row][col + start_col] = self.minPlayer
                        next_board_index = 3*row + col
                        bestValue = min(bestValue, self.alphabeta_designed(depth + 1, next_board_index, alpha, beta, not isDesigned))
                        # Undo the move
                        if bestValue < beta:
                            self.board[row + start_row][col + start_col] = '_'
                            return bestValue
                        else:
                            alpha = bestValue
                        self.board[row + start_row][col + start_col] = '_'
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
                        bestValue = max(bestValue, self.alphabeta_designed(depth + 1, next_board_index, alpha, beta, not isDesigned))
                        if bestValue > alpha:
                            self.board[row + start_row][col + start_col] = '_'
                            return bestValue
                        else:
                            beta = bestValue
                        # Undo the move
                        self.board[row + start_row][col + start_col] = '_'
            return bestValue
        return bestValue

    def minimax_designed(self, depth, currBoardIdx, isDesigned):
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
            return self.evaluateDesigned(currBoardIdx)


        start_row, start_col = self.globalIdx[currBoardIdx]

        # If this is maxPlayer
        if isDesigned:
            bestValue = inf
            # Traverse all cells
            for row in range(3):
                for col in range(3):
                    if self.board[row + start_row][col + start_col] == '_':
                        self.board[row + start_row][col + start_col] = self.minPlayer
                        # If either player wins, return best score
                       # pdb.set_trace()
                        next_board_index = 3*row + col
                        curr_value = self.minimax_designed(depth + 1, next_board_index,not isDesigned)
                        if curr_value < bestValue:
                            bestValue = curr_value
                        #self.board[row + start_row][col + start_col] == '_'
                      #  pdb.set_trace()
                        # Undo the move
                        self.board[row + start_row][col + start_col] = '_'
            return bestValue
        else:
            bestValue = inf
            # Traverse all cells
            for row in range(3):
                for col in range(3):
                    if self.board[row + start_row][col + start_col] == '_':
                        self.board[row + start_row][col + start_col] = self.maxPlayer
                        next_board_index = 3*row + col
                        # bestValue = min(bestValue, self.minimax(depth + 1, next_board_index, not isMax))
                        # Undo the move
                        curr_value = self.minimax_designed(depth + 1, next_board_index, not isDesigned)
                        if curr_value > bestValue:
                            bestValue = curr_value
                        self.board[row + start_row][col + start_col] = '_'
            return bestValue
        return bestValue

    def find_best_move_designed(self, currBoardIdx, isDesigned, isMinimax):
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        currBoardIdx(int): current local board index
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        isMinimax(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm
        output:
        best_move(tuple):the coordinates of the best move
        """
        # pdb.set_trace()

        start_row, start_col = self.globalIdx[currBoardIdx]
        best_move = (0, 0)
        if isDesigned:
            player = self.minPlayer
            opponent = self.maxPlayer
            bestValue = -inf
            for row in range(3):
                for col in range(3):
                    if self.board[row + start_row][col + start_col] == '_':
                        self.board[row + start_row][col + start_col] = player
                        if isMinimax:
                            move_value = self.minimax_designed(1, currBoardIdx, isDesigned)
                        else:
                            move_value = self.alphabeta_designed(1, currBoardIdx, -inf, inf, isDesigned)
                        if move_value > bestValue:
                            best_move = (row, col)
                            bestValue = move_value
                        self.board[row + start_row][col + start_col] = '_'
        else:
            player = self.maxPlayer
            opponent = self.minPlayer
            bestValue = inf
            for row in range(3):
                for col in range(3):
                    if self.board[row + start_row][col + start_col] == '_':
                        self.board[row + start_row][col + start_col] = player
                        if isMinimax:
                            move_value = self.minimax_designed(1, currBoardIdx, isDesigned)
                        else:
                            move_value = self.alphabeta_designed(1, currBoardIdx, -inf, inf, isDesigned)
                        if move_value < bestValue:
                            best_move = (row, col)
                            bestValue = move_value
                        self.board[row + start_row][col + start_col] = '_'
        return best_move

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
        bestValue=[]
        gameBoards=[]
        winner=0
        expandedNodes = 0

        currBoardIdx = self.startBoardIdx
        i = 0
        isDesigned = True

        while self.checkMovesLeft():

            if i%1 == 0:
                self.printGameBoard()
                print(self.checkWinner())
            if isDesigned: #max first sequence
                best_move = self.find_best_move_designed(currBoardIdx, True,True)
                self.make_move(best_move[0], best_move[1],currBoardIdx,False)
                #update result arrays
                if self.checkWinner() != 0:
                    winner = self.checkWinner()
                    break
                currBoardIdx = 3*best_move[0]+best_move[1]

                isDesigned = not isDesigned
            else: #min first sequence
                best_move = self.find_best_move_designed(currBoardIdx,True,True)
                self.make_move(best_move[0],best_move[1],currBoardIdx,True)
                if self.checkWinner() !=0:
                    winner = self.checkWinner()
                    break
                currBoardIdx = 3*best_move[0] + best_move[1]
                isDesigned = not isDesigned

            i += 1
        return gameBoards, bestMove, winner

    def _AIMove(self, currBoardIdx):
        """
        Use the designed agent to move.
        Call function find_best_move_designed()
        Arguments:
            currBoardIdx: int -- The subBoard that should be modified.
        Returns:
            nextBoardIndex -- The index of the next board.
            moveCoord -- The world coordinates of the move.
        Raises:
            ValueError -- The AI moves to a invalid position.
        """
        startRow, startCol = self.globalIdx[currBoardIdx]
        # Rely on the self.find_best_move_designed() function to make the correct move.
        bestMove = self.find_best_move_designed(currBoardIdx, True, True)
        r = bestMove[0] + startRow
        c = bestMove[1] + startCol
        if self.board[r][c] != "_":
            print("--------Invalid move by AI--------")
            raise ValueError
        self.board[r][c] = "X"
        nextBoardIndex = 3*(r%3)+(c%3)
        moveCoord = (r,c)
        return nextBoardIndex, moveCoord

    def _HumanMove(self, currBoardIdx):
        """
        Modify the self.board once with the human input.
        Arguments:
            currBoardIdx: int -- The subBoard that should be modified.
        Returns:
            nextBoardIndex -- The index of the next board.
            moveCoord -- The world coordinates of the move.
        """
        startRow, startCol = self.globalIdx[currBoardIdx]
        while True:
            print("Valid row are {}-{}".format(startRow, startRow+3))
            print("Valid col are {}-{}".format(startCol, startCol+3))
            rowIndex = input("Row index: ")
            try:
                rowIndex = int(rowIndex)
            except ValueError:
                print("Invalid rowIndex input")
                continue

            colIndex = input("Col index: ")
            try:
                colIndex = int(colIndex)
            except ValueError:
                print("Invalid colIndex input")
                continue

            if rowIndex >= startRow and rowIndex < startRow+3 and colIndex >= startCol and colIndex < startCol+3:
                if self.board[rowIndex][colIndex] == "_":
                    self.board[rowIndex][colIndex] = "O"
                    nextBoardIndex = 3*(rowIndex%3)+(colIndex%3)
                    moveCoord = (rowIndex, colIndex)
                    return nextBoardIndex, moveCoord
                else:
                    print("Occupied position!")
                    print()
            else:
                print("Invalid Board!")
                print()

    def playGameHuman(self):
        """
        This function implements the processes of the game of your own agent vs a human.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.

        Implemented:
        bestMove: list of tuples - The world coordinates of each step for both human and agent.
        gameBoards: list of 2d lists - The game board at the end of each round.
        """
        #YOUR CODE HERE
        bestMove=[]
        gameBoards=[]
        winner=0
        currBoardIdx = self.startBoardIdx
        while self.checkMovesLeft() == True:
            currBoardIdx, move = self._AIMove(currBoardIdx)
            bestMove.append(move)
            if self.checkWinner() != 0:
                print("------Game over, you lost!------")
                break

            self.printGameBoard()

            currBoardIdx, move = self._HumanMove(currBoardIdx)
            bestMove.append(move)
            if self.checkWinner() != 0:
                print("-------Congraduation!-------")
                break
            gameBoards.append(copy.deepcopy(self.board))
        return gameBoards, bestMove, winner

# -----------Tests------------#
def testHuman():
    uttt = ultimateTicTacToe()
    uttt.playGameHuman()

def testBoard():
    uttt = ultimateTicTacToe()
    uttt.board =   [['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','O','X','O','_','_','_'],
                    ['_','_','_','_','O','_','_','_','_'],
                    ['_','_','_','_','X','X','X','_','X'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_']]
    return

def testDesigned():
    uttt = ultimateTicTacToe()
    gameBoards, bestMove, winner = uttt.playGameYourAgent()
    uttt.printGameBoard()

    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")

def testPredefined():
    uttt = ultimateTicTacToe()
    gameBoards, bestMove, expandedNodes, bestValue, winner = uttt.playGamePredifinedAgent(True,True,True)
    uttt.printGameBoard()

    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")

def testMyPredefined():
    uttt = ultimateTicTacToe()
    # myPlayGamePredifinedAgent(maxFirst?, isMinimaxOffensive, isMinimaxDefensive)
    gameBoards, bestMove, expandedNodes, bestValue, winner = uttt.myPlayGamePredifinedAgent(True,True,True)
    uttt.printGameBoard()

    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")
#  -------------------------- #

if __name__=="__main__":
    # testPredefined()
    testMyPredefined()

    # print(uttt.evaluatePredifined(False))
    # print(uttt.evaluatePredifined(True))
