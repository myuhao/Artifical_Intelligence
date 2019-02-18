import numpy as np;


class DLX:

    class Node:
        def __init__(self, left=None, right=None, up=None, down=None):
            self.l = left # Type = Node.
            self.r = right
            self.u = up
            self.d = down
            self.column = None # Point to the column header node that identify the node's column label
            self.rowID = None # Type = int, 0-indexed, padded row number.
            self.colID = None # Type = int, 0-indexed non-padded col number.
            self.nodeCt = 0 # Type = int, Number of nodes in the column.

        def print(self):
            print("Node rowID is {}, colID is {}".format(self.rowID, self.colID))
            print()

    def __init__(self, A):
        '''
        Ctor. A is the input np matrix. The first row is pedded with np.ones((nCol,))
        '''
        self.header = self._convertMatToHeader(A)


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

                    # Point the node to its neighbors.
                    # --------------------------------------------------
                    # Right
                    currColIndex = j
                    while True:
                        currColIndex = moveRight(currColIndex)
                        if currColIndex == j or A[i][currColIndex] == 1:
                            break
                    matrix[i][j].r = matrix[i][currColIndex]
                    # Left
                    currColIndex = j
                    while True:
                        currColIndex = moveLeft(currColIndex)
                        if currColIndex == j or A[i][currColIndex] == 1:
                            break
                    matrix[i][j].l = matrix[i][currColIndex]
                    # Down
                    currRowIndex = i
                    while True:
                        currRowIndex = moveDown(currRowIndex)
                        if currRowIndex == i or A[currRowIndex][j] == 1:
                            break
                    matrix[i][j].d = matrix[currRowIndex][j]
                    # Up
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
        header.r = matrix[0][0]
        header.l = matrix[0][nCol-1]
        matrix[0][0].l = header
        matrix[0][nCol-1].r = header
        return header

    def print(self):
        # print(self.header.r.nodeCt)
        curr = self.header
        for i in range(10):
            curr.print()
            curr = curr.r


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


    solution = DLX(MAT)
    solution.print()


if __name__ == "__main__":
    test()
