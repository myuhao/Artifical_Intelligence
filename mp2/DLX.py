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
                if A[i][j] == 1:
                    # Increment number of node at the column node.
                    if i != 0:
                        matrix[0][j].nodeCt += 1
                    # Point the node back to its column header.
                    matrix[i][j].column = matrix[0][j]
                    # Update the node's rowID and colID
                    matrix[i][j].rowID = i
                    matrix[i][j].colID = j

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
        # Base case: we covered all col.
        if self.header.r == self.header:
            print("Solution is {}".format(self.solution))
            self.SOL.append(self.solution.copy())
            # yield self.solution.copy() # NOT working!!!
            return

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
            self.solve()
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

def test():
    rowHeader = np.ones((7,))
    # rowHeader =   [1, 1, 1, 1, 1, 1, 1]
    row0 = np.array([1, 0, 0, 1, 0, 0, 1])
    row1 = np.array([1, 0, 0, 1, 0, 0, 0])
    row2 = np.array([0, 0, 0, 1, 1, 0, 1])
    row3 = np.array([0, 0, 1, 0, 1, 1, 0])
    row4 = np.array([0, 1, 1, 0, 0, 1, 1])
    row5 = np.array([0, 1, 0, 0, 0, 0, 1])
    row6 = np.array([1, 0, 0, 1, 0, 0, 0])
    MAT = np.array([rowHeader, row0, row1, row2, row3, row4, row5, row6]) # Ans = [6, 4, 2] [6, 4, 7] (1-indexed)

    # rowHeader = np.ones((7,))
    # # rowHeader =   [1, 1, 1, 1, 1, 1, 1]
    # row0 = np.array([0, 0, 1, 0, 1, 1, 0])
    # row1 = np.array([1, 0, 0, 1, 0, 0, 1])
    # row2 = np.array([0, 1, 1, 0, 0, 1, 0])
    # row3 = np.array([1, 0, 0, 1, 0, 0, 0])
    # row4 = np.array([0, 1, 0, 0, 0, 0, 1])
    # row5 = np.array([0, 0, 0, 1, 1, 0, 1])
    # MAT = np.array([rowHeader, row0, row1, row2, row3, row4, row5]) # Ans = [6, 4, 2] [6, 4, 7] (1-indexed)


    A = np.genfromtxt("matrix.csv", delimiter=",")
    A = A[1::,1::]
    A = np.insert(A, 0, np.ones((A.shape[1],)), axis=0)

    A = MAT
    solution = DLX(A)
    # solution._cover(solution.header.r.d)
    # print(solution.shape)
    ct = 0
    solution.solve()
    # for i in solution.solve():
    #     print(i)

    print(solution.SOL)

class CONST:
    dominos = [np.array([[i],[i]]) for i in range(1,31)]
    triominos = [np.array([[i,0],[i,i]]) for i in range(1,21)]
    # List of petnominos.
    petnominos = [np.array([[0,1,1],
                       [1,1,0],
                       [0,1,0]]), #F
            np.array([[2],
                      [2],
                      [2],
                      [2],
                      [2]]), #I
            np.array([[3,0],
                      [3,0],
                      [3,0],
                      [3,3]]), #L
            np.array([[0,4],
                      [0,4],
                      [4,4],
                      [4,0]]), #N
            np.array([[5,5],
                      [5,5],
                      [5,0]]), #P
            np.array([[6,6,6],
                      [0,6,0],
                      [0,6,0]]), #T
            np.array([[7,0,7],
                      [7,7,7]]), #U
            np.array([[8,0,0],
                      [8,0,0],
                      [8,8,8]]), #V
            np.array([[9,0,0],
                      [9,9,0],
                      [0,9,9]]), #W
            np.array([[0,10,0],
                      [10,10,10],
                      [0,10,0]]), #X
            np.array([[0,11],
                      [11,11],
                      [0,11],
                      [0,11]]), #Y
            np.array([[12,12,0],
                      [0,12,0],
                      [0,12,12]])] #Z

    board_6x10 = np.ones((6,10))
    board_5x12 = np.ones((5,12))
    board_3x20 = np.ones((3,20))
    empty_chessboard = np.ones((8,8))
    empty_chessboard[3][3] = empty_chessboard[3][4] = empty_chessboard[4][3]  = empty_chessboard[4][4] = 0

class Converter:
    def __init__(self, board, pents):
        self.board = board
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
        self.npRow2Pent = {} # Use np.row -> pent. First 60 columns.

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
                if self.board[i,j] == 0:
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
                            if copy_board[h[0], h[1]] != 0:
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
                        self.npRow2Pent[flattened.tostring()] = int(pentValue)
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

    def print(self):
        print(self.matrix)


def testNumOrentation():
    """
    Test the number of orentations by checking with the reference answers
    <http://www.basic.northwestern.edu/g-buehler/pentominoes/speech.htm>
    """
    ANS = {
        "F": 8,
        "L": 8,
        "N": 8,
        "P": 8,
        "Y": 8,
        "T": 4,
        "U": 4,
        "V": 4,
        "W": 4,
        "Z": 4,
        "I": 2,
        "X": 1
    }

    letters = [
        "F",
        "I",
        "L",
        "N",
        "P",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z"
    ]

    index2ANS = {}
    for i in range(12):
        index2ANS[i] = ANS[letters[i]]

    A = Converter(CONST.board_6x10, CONST.petnominos)
    for i in range(len(CONST.petnominos)):
        print(len(A._getAllOrentation(CONST.petnominos[i])) == index2ANS[i])

def testAllPositions():
    A = Converter(CONST.empty_chessboard, CONST.petnominos)
    for i in A._getAllPosition(CONST.petnominos[3]):
        # print(i)
        if i != None:
            if np.sum(i) != 5:
                print(i)
    for i in A.npRow2Pent:
        print(A.npRow2Pent[i])

def testFinalMatrix():
    A = Converter(CONST.board_6x10, CONST.petnominos)
    print(A.getMatrix().shape)

def test6x10SolNum():
    '''
    Expect: 9356 solutions
    Accutal: 9356 solutions
    Save answer to a new CSV next time.
    '''
    A = Converter(CONST.board_6x10, CONST.petnominos)
    mat = A.getMatrix()
    DLL = DLX(mat)
    DLL.solve()
    for i in DLL.SOL:
        print(i)

def test3x20():
    '''
    Expect: 8 solutions
    Accutal: 8  solutions
    Save answer to a new CSV next time.
    '''
    A = Converter(CONST.board_3x20, CONST.petnominos)
    mat = A.getMatrix()
    DLL = DLX(mat)
    DLL.solve()
    for i in DLL.SOL:
        print(i)
    print("Number of solutions is {}".format(len(DLL.SOL)))

def testHoles():
    '''
    Expect: 520? solutions
    Acctual: 520 solutions
    '''
    A = Converter(CONST.empty_chessboard, CONST.petnominos)
    mat = A.getMatrix()
    DLL = DLX(mat)
    DLL.solve()
    for i in DLL.SOL:
        print(i)
    print("Number of solutions is {}".format(len(DLL.SOL)))

def testAll():
    A = Converter(CONST.board_6x10, CONST.petnominos)
    mat = A.getMatrix()
    DLL = DLX(mat)
    # for i in range(mat.shape[1]):
    #     print(np.max(mat[:,i]) == 1)
    # for i in range(mat.shape[0]):
    #     print(np.sum(mat[i,:]) == 6)

if __name__ == "__main__":
    testHoles()

