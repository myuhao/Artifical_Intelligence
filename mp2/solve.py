# -*- coding: utf-8 -*-
import numpy as np
import sys # Use sys.maxsize

class DLX:

    class Node:
        def __init__(self, left=None, right=None, up=None, down=None):
            self.l = left # Type = Node, left node.
            self.r = right
            self.u = up
            self.d = down
            self.column = None # Point to the column header node that identify the node's column label
            self.rowID = None # Type = int, 0-indexed, padded row number.
            self.colID = None # Type = int, 0-indexed non-padded col number.
            self.nodeCt = 0 # Type = int, Number of nodes in the column.
            self.val = 0 # Type = int, the value of the matrix element.

        def print(self):
            print("Node rowID and colID is ({}, {}).".format(self.rowID, self.colID))

    # ------------ End of Clss Node -------------

    def __init__(self, A):
        '''
        Ctor. A is the input np matrix. The first row is pedded with np.ones((nCol,))
        '''
        self.header = self._convertMatToHeader(A)
        self.shape = A.shape
        self.solution = []
        self.SOL = []

    def _convertMatToHeader(self, A):
        """
        Convert the np matrix to a DLX strcuture.
        Return the header Node.
        """
        nRow, nCol = A.shape

        # Four helper functions to help row over the indice.
        def moveRight(currColIndex):
            return (currColIndex+1)%nCol
        def moveLeft(currColIndex):
            return nCol-1 if currColIndex-1<0 else currColIndex-1
        def moveDown(currRowIndex):
            return (currRowIndex+1)%nRow
        def moveUp(currRowIndex):
            return nRow-1 if currRowIndex-1<0 else currRowIndex-1

        # Constructing a matrix with nodes.
        # First row are column nodes, therefore row is 1-indexed.
        matrix = []
        for i in range(nRow):
            tempRow = []
            for j in range(nCol):
                tempRow.append(self.Node())
            matrix.append(tempRow)

        # Iterate over the input matrix A to collect all "1".
        for i in range(nRow):
            for j in range(nCol):
                # How about count all matrix element that is larger than 1?
                if A[i][j] >= 1:
                    # Increment number of node at the column node.
                    if i != 0:
                        matrix[0][j].nodeCt += 1
                    # Point the node back to its column header.
                    matrix[i][j].column = matrix[0][j]
                    # Update the node's rowID and colID
                    matrix[i][j].rowID = i
                    matrix[i][j].colID = j
                    # Update the node's value
                    matrix[i][j].val = A[i][j]

                    # Get the next 1 and have current node point to it.
                    # --------------------------------------------------
                    # Right neighbor.
                    currColIndex = j
                    while True:
                        currColIndex = moveRight(currColIndex)
                        if currColIndex == j or A[i][currColIndex] == 1:
                            break
                    matrix[i][j].r = matrix[i][currColIndex]
                    # Left neighbor.
                    currColIndex = j
                    while True:
                        currColIndex = moveLeft(currColIndex)
                        if currColIndex == j or A[i][currColIndex] == 1:
                            break
                    matrix[i][j].l = matrix[i][currColIndex]
                    # Down neighbor.
                    currRowIndex = i
                    while True:
                        currRowIndex = moveDown(currRowIndex)
                        if currRowIndex == i or A[currRowIndex][j] == 1:
                            break
                    matrix[i][j].d = matrix[currRowIndex][j]
                    # Up neighbor.
                    currRowIndex = i
                    while True:
                        currRowIndex = moveUp(currRowIndex)
                        if currRowIndex == i or A[currRowIndex][j] == 1:
                            break
                    matrix[i][j].u = matrix[currRowIndex][j]
                    # --------------------------------------------------
                    # print("({},{}) is linked to ({}, {})".format(i,j,i,currColIndex))
        # Connect the header node.
        header = self.Node()
        header.nodeCt = sys.maxsize
        header.r = matrix[0][0]
        header.l = matrix[0][nCol-1]
        matrix[0][0].l = header
        matrix[0][nCol-1].r = header
        return header

    def _getMinColHeaderNode(self):
        '''
        Get the column header of the column with the smallest number
        of node.
        If nodeCt is the same, use the column with lower column index.
        Returns:
            Node -- The column header that has the lowest nodeCt.
        Raises:
            IndexError -- The matrix is empty.
        '''
        currNode = self.header.r
        minNode = currNode
        if currNode == None:
            print("ERROR: EMPTY MATRIX.")
            raise IndexError
        while currNode != self.header:
            minNode = minNode if minNode.nodeCt<=currNode.nodeCt else currNode
            currNode = currNode.r
        return minNode

    def _cover(self, nNode):
        # Remove the column header first.
        colHeader = nNode.column
        colHeader.r.l = colHeader.l
        colHeader.l.r = colHeader.r

        # Move down the linked list and remove rows.
        # Initialized at the first node that is not the column header.
        currRow = colHeader.d
        # We will move down this column
        while currRow != nNode.column:
            # When encountered a node, we will move right.
            currCol = currRow.r
            while currCol != currRow:
                # When encountered a node, detached it from the column.
                currCol.d.u = currCol.u
                currCol.u.d = currCol.d
                # Decreament the nodeCt.
                currCol.column.nodeCt -= 1

            # Increament the currNode.
                currCol = currCol.r
            currRow = currRow.d
        return

    def _uncover(self, nNode):
        colHeader = nNode.column
        # Move up the linked list and "reconnect" the nodes.
        # Initialize to the first UP node.
        currRow = colHeader.u
        # We will move up.
        while currRow != nNode.column:
            # We will move left.
            currCol = currRow.l
            while currCol != currRow:
                # Reconnect.
                currCol.u.d = currCol
                currCol.d.u = currCol
                # Increment the nodeCt
                currCol.column.nodeCt += 1
            # Move the current node in the opposite direction.
                currCol = currCol.l
            currRow = currRow.u

        # Reconnect the column headers.
        colHeader.l.r = colHeader
        colHeader.r.l = colHeader
        return

    def solve(self):
        '''
        Function to solve the exact cover problem using
        algorithm X and DLX structure.
        Terminate when one solution is found.
        Returns:
            bool -- If a solution has been found.
        '''
        # Base case: we covered all col.
        if self.header.r == self.header:
            print("Solution is {}".format(self.solution))
            self.SOL.append(self.solution.copy())
            # yield self.solution.copy() # NOT working!!!
            return True

        # Choose the column to cover.
        minCol = self._getMinColHeaderNode()
        # Cover it.
        self._cover(minCol)

        # Move down the column and add the row index to the solution.
        currRow = minCol.d
        while currRow != minCol:
            # print(currRow.rowID)
            self.solution.append(currRow.rowID)
            # Handle the other "1"s by moving right.
            currCol = currRow.r
            while currCol != currRow:
                self._cover(currCol)
                # Move right, inner loop.
                currCol = currCol.r

            # Recursively solve the current DLX mesh again.
            if self.solve() == True:
                return True
            # Trackback step.
            self.solution.pop()

            minCol = currRow.column
            currCol = currRow.l
            while currCol != currRow:
                self._uncover(currCol)
                currCol = currCol.l

            # Move down, outer loop.
            currRow = currRow.d
        self._uncover(minCol)


    def print(self):
        currNode = self.header.r
        while currNode != self.header:
            currNode.print()
            print(currNode.nodeCt)
            currNode = currNode.r

    def convertToMatrix(self):
        mat = np.zeros(self.shape)
        mat[0,:] = np.ones((self.shape[1],))
        currColHeader = self.header.r
        for i in range(self.shape[1]-1):
            currNode = currColHeader.d
            while currNode != currColHeader:
                currNode.print()
                mat[currNode.rowID, currNode.colID] = 1
                currNode = currNode.d
            currColHeader = currColHeader.r
        return mat
# ----------- End of class DLX ------------

class Converter:
    def __init__(self, board, pents):
        self.HOLEVAL = 2019 # Randomly choosen to represent hole.
        self.board = self._fillAllHoles(board, self.HOLEVAL)
        self.pents = pents
        self.nCol = 0 # Number of columns.
        for i in self.board:
            for j in i:
                if j == 1:
                    self.nCol += 1
        self.nCol += len(pents)
        self.matrix = np.ones((1, self.nCol))
        self.nRow = 1
        self.rIdx2Pent = {} # Row index of the matrix -> pent.
        '''
        Use np.row -> pent.
        Key: tostring(row[0:60]), first 60 columns.
        Val: (pi, (rowi, coli))
        '''
        self.npRow2Pent = {}

    # 0-based np.matrix index <-> int.
    def _twoD2oneD(self, ridx, cidx):
        return ridx*self.nCol + cidx
    def _oneD2twoD(self, idx):
        return (idx//self.nCol, idx%self.nCol)

    def _getAllOrentation(self, pent):
        '''
        Get a list of all possible orentation of a single pentomino

        Returned as a list.
        '''
        allOrentation = [pent]

        def isInOren(p):
            for i in allOrentation:
                if np.all(i == p) and i.shape == p.shape: # "I" shaped pent needs special considerations.
                    return True
            return False

        for flipAxis in range(4):
            flipped = pent.copy()
            if flipAxis == 2:
                flipped = pent.copy()
            elif flipAxis == 3:
                flipped = np.flip(pent,0)
                flipped = np.flip(flipped,1)
            else:
                flipped = np.flip(pent, flipAxis)
            for rotTimes in range(4):
                newPent = np.rot90(flipped, k=rotTimes)
                if not isInOren(newPent):
                    allOrentation.append(newPent)
        return allOrentation

    def _isValidPosition(self, pent, ridx, cidx):
        pass

    def _fillAllHoles(self, board, fillVal):
        copy_board = board.copy()
        for i in range(len(copy_board)):
            for j in range(len(copy_board[i])):
                if copy_board[i][j] == 0:
                    copy_board[i][j] = fillVal
        return copy_board

    def _getAllPosition(self, pent):
        """
        Pass in a pent as np.array and return a list of np.array to be
        inserted in to the final matrix.
        Arguments:
            pent np.array -- A pentominoe to be placed to the borad.
        Yield:
            A list of np.array of shape (1,nCol) that needs to be inserted
            to the matrix.
        """
        # If rectanglar, holes will not be a problem.
        holes = [] # List of tuples of the "0"s in the board.
        holeIdx = []
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i,j] == self.HOLEVAL:
                    holes.append((i,j))
                    holeIdx.append(self.board.shape[0]*i+j)
        '''
        First add the value of the pent to the board if possible.
        Then check if "0"/holes changed value.
        Then flattened the array with np.flatten().
        Then delete the holes with np.delete().
        Then add the pent index to the front.
        '''
        resultList = []
        # Iterate over every possible orentations
        for pi in self._getAllOrentation(pent):
            prow, pcol = pi.shape
            # Add the board value to the current board first.
            for i in range(self.board.shape[0] + 1 - prow):
                for j in range(self.board.shape[1] + 1 - pcol):
                    isInTheHole = False # bool to check if the
                                        # pent intersect the hole
                    copy_board = self.board.copy()
                    copy_board[i:i+prow, j:j+pcol] += pi
                    # Check for holes.
                    if len(holes) != 0:
                        for h in holes:
                            if copy_board[h[0], h[1]] != self.HOLEVAL:
                                # This means something is added to the hole.
                                # So we do not add this position.
                                isInTheHole = True
                    if isInTheHole == False:
                        flattened = copy_board.flatten()
                        flattened = np.delete(flattened, holeIdx)
                        # Renormalized to 1.
                        flattened -= 1
                        pentValue = flattened[np.nonzero(flattened)][0]
                        flattened /= pentValue
                        # Handle collision.
                        if flattened.tostring() in self.npRow2Pent.keys():
                            print("-----Key already exists-----")
                            raise KeyError
                        # Hashing using np.tostring().
                        # Generate tuple of (pi, (rowi,coli))
                        tileANS = (pi.copy(), (i,j))
                        self.npRow2Pent[flattened.tostring()] = tileANS
                        yield flattened
                        resultList.append(flattened)
        # return resultList

    def getMatrix(self):
        pent_idx = 0
        for pi in self.pents:
            for row in self._getAllPosition(pi):
                newRow = row.copy()
                IDRow = np.zeros((1, len(self.pents)))
                IDRow[0,pent_idx] = 1
                newRow = np.append(IDRow, newRow)
                newRow = newRow.reshape((1, self.nCol))
                self.matrix = np.append(self.matrix, newRow, axis=0)
            pent_idx += 1
        return self.matrix

    def getPent(self, ridx):
        row = self.matrix[ridx, len(self.pents)::]
        key = row.tostring()
        return self.npRow2Pent[key]

    def print(self):
        print(self.matrix)

    def printDict(self):
        for k in self.npRow2Pent:
            print(k)
            break
# -------- End of class Converter ---------

def solve(board, pents):
    """
    This is the function you will implement. It will take in a numpy array of the board
    as well as a list of n tiles in the form of numpy arrays. The solution returned
    is of the form [(p1, (row1, col1))...(pn,  (rown, coln))]
    where pi is a tile (may be rotated or flipped), and (rowi, coli) is
    the coordinate of the upper left corner of pi in the board (lowest row and column index
    that the tile covers).

    -Use np.flip and np.rot90 to manipulate pentominos.

    -You can assume there will always be a solution.
    """

    solution = []
    cvt = Converter(board, pents)
    matrix = cvt.getMatrix()
    solver = DLX(matrix)
    solver.solve()
    for ans in solver.SOL:
        for ridx in ans:
            solution.append(cvt.getPent(ridx))
        break

    return solution
