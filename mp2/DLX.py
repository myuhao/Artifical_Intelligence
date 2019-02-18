import numpy as np;
import sys;


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
    # rowHeader = np.ones((7,))
    # # rowHeader =   [1, 1, 1, 1, 1, 1, 1]
    # row0 = np.array([1, 0, 0, 1, 0, 0, 1])
    # row1 = np.array([1, 0, 0, 1, 0, 0, 0])
    # row2 = np.array([0, 0, 0, 1, 1, 0, 1])
    # row3 = np.array([0, 0, 1, 0, 1, 1, 0])
    # row4 = np.array([0, 1, 1, 0, 0, 1, 1])
    # row5 = np.array([0, 1, 0, 0, 0, 0, 1])
    # row6 = np.array([1, 0, 0, 1, 0, 0, 0])
    # MAT = np.array([rowHeader, row0, row1, row2, row3, row4, row5, row6]) # Ans = [6, 4, 2] [6, 4, 7] (1-indexed)

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


    solution = DLX(A)
    # solution._cover(solution.header.r.d)
    # print(solution.shape)
    solution.solve()
    # print(solution.solution)

if __name__ == "__main__":
    import time
    t0 = time.time()
    test()
    print("Running time is {0:.3f}s.".format((time.time()-t0)/60))
