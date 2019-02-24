import numpy as np;


class LabelMatrix:

    def __init__(self, A):
        self.matrix = A
        self.shape = A.shape
        self.dataFrame = self.toLabel(A)

    def print(self):
        print(self.dataFrame)
        return;

    def toLabel(self, A):
        """
        Add index in to a numpy array.

        Arguments:
            A {ndarray} -- The original matrix
        Returns:
            ndarray -- Indexed matrix. Index is at the first row and col.
        """
        row, col = self.shape
        row += 1
        col += 1

        newMat = np.ones((row, col))
        newMat *= -1

        newMat[0:row-1, 0:col-1] = A.copy()
        newMat[row-1,:] *= np.arange(0, col)
        newMat[:,col-1] *= np.arange(0, row)
        newMat[-1, -1] = 2
        return newMat

    def getRowByLabel(self, rlabel):
        """ Use the label to retrive the row."""
        for label in range(len(dataFrame[:,-1])):
            if dataFrame[label,-1] == rlabel:
                return dataFrame[label,:]

        raise IndexError

    def getColByLabel(self, clabel):
        for label in range(len(dataFrame[-1,:])):
            if dataFrame[-1, label] == clabel:
                return dataFrame[:,label]

        raise IndexError

    def deleteRowByLabel(self, rlabel)


if __name__ == "__main__":
    # Given a 2D numpy array of 1 and 0, get its exact cover rows
    row0 = np.array([0, 0, 1, 0, 1, 1, 0])
    row1 = np.array([1, 0, 0, 1, 0, 0, 1])
    row2 = np.array([0, 1, 1, 0, 0, 1, 0])
    row3 = np.array([1, 0, 0, 1, 0, 0, 0])
    row4 = np.array([0, 1, 0, 0, 0, 0, 1])
    row5 = np.array([0, 0, 0, 1, 1, 0, 1])
    MAT = np.array([row0, row1, row2, row3, row4, row5])
    MAT = LabelMatrix(MAT)
    MAT.print()
