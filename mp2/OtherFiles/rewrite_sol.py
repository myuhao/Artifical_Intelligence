import numpy as np

def addIndex(A):
    """
    Add index in to a numpy array.

    Arguments:
        A {ndarray} -- The original matrix
    Returns:
        ndarray -- Indexed matrix. Index is at the first row and col.
    """
    row, col = A.shape
    row += 1
    col += 1

    newMat = np.ones((row, col))
    newMat *= -1
    # newMat[1:row, 1:col] = A.copy()
    # newMat[0,:] = np.arange(-1,col-1)
    # newMat[:,0] = np.arange(-1,row-1)

    newMat[0:row-1, 0:col-1] = A.copy()
    newMat[row-1,:] *= np.arange(0, col)
    newMat[:,col-1] *= np.arange(0, row)
    newMat[-1, -1] = 2
    return newMat

def deleteRowByIdx(A, idx):
    row, col = A.shape
    for rowIdx in range(len(A[:,-1])):
        if A[rowIdx,-1] == idx:
            return np.delete(A, (rowIdx), axis=0)
    print("___Index {} is not found in A___".format(idx))
    raise IndexError

def deleteColByIdx(A, idx):
    row, col = A.shape
    for colIdx in range(len(A[-1,:])):
        if A[-1,colIdx] == idx:
            return np.delete(A, (colIdx), axis=1)
    print("___Index {} is not found in A___".format(idx))
    raise IndexError

def getColIndexFromLabel(A, cidx):
    for i in range(len(A[-1,:])):
        if A[-1,i] == cidx:
            return i
    raise IndexError

def solveMatrix(A):
    '''
    Get the row indices of the exact cover.
    Arguments:
        A {np.ndarray} -- Input matrix
    '''
    # If there is no column, end.
    # Take the index column into consideration.
    if A.shape[1] <= 1:
        yield []
    else:
        NUM_ROW, NUM_COL = A.shape
        # Start from the column c with the fewest "1"s.
        temp = A[0:-1, 0:-1]
        c = temp.sum(axis=0).argmin() # This is the index of the column.

        # Try each row that has "1".
        for r in range(len(A[:,c])):
            ansIndex = A[r,-1]
            # Skip row that is not "1".
            if A[r,c] != 1:
                continue

            B = A.copy()

            # Now iterate over row r and get col index that is "1".
            for j in range(len(A[r,:])):
                if A[r,j] == 1: # j now is the col index with "1".
                    # First delete the rows with ones in this col

                    print(j)
                    # convert j (index in A) to label to (index in B)
                    label = A[-1,j]
                    print(label)
                    idx = getColIndexFromLabel(B, label)
                    B = B[B[:,idx] != 1]
                    print(B)
                    # Delete the column
                    B = deleteColByIdx(B, A[-1,j])
                    print(B)
                    print("_____New loop_____")

                # Skip col that is not "1":
                if A[r,j] != 1:
                    continue

            for sol in solveMatrix(B):
                yield [ansIndex] + sol




if __name__ == "__main__":
    # Given a 2D numpy array of 1 and 0, get its exact cover rows
    row0 = np.array([0, 0, 1, 0, 1, 1, 0])
    row1 = np.array([1, 0, 0, 1, 0, 0, 1])
    row2 = np.array([0, 1, 1, 0, 0, 1, 0])
    row3 = np.array([1, 0, 0, 1, 0, 0, 0])
    row4 = np.array([0, 1, 0, 0, 0, 0, 1])
    row5 = np.array([0, 0, 0, 1, 1, 0, 1])
    MAT = np.array([row0, row1, row2, row3, row4, row5])
    MAT = MAT[MAT[:,0] != 1]
    print(MAT)

    row0 = np.array([0, 0, 1, 0, 1, 1, 0])
    row1 = np.array([1, 0, 0, 1, 0, 0, 1])
    row2 = np.array([0, 1, 1, 0, 0, 1, 0])
    row3 = np.array([1, 0, 0, 1, 0, 0, 0])
    row4 = np.array([0, 1, 0, 0, 0, 0, 1])
    row5 = np.array([0, 0, 0, 1, 1, 0, 1])
    MAT = np.array([row0, row1, row2, row3, row4, row5])
    MAT = addIndex(MAT)

    for i in solveMatrix(MAT):
        print(i)



class DateFrame:
    def __init__(self, A):
        self.mat
        self.col
        self.row

    def deleteRow(self, ridx):
        pass

    def deleteCol(self, cidx):
        pass

    def getByIndex(self, ridx, cidx):
        pass

    def getRowByIndex(self, ridx):
        pass

    def getColByIndex(self, cidx):
        pass

    def iterateOverIndex(self):
        pass
