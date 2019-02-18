# -*- coding: utf-8 -*-
import numpy as np

def addLabel(A):
    """
    Add label in to a numpy array.

    Arguments:
        A {ndarray} -- The original matrix
    Returns:
        ndarray -- Labeled matrix. Index is at the last row and col.
    """
    row, col = A.shape
    row += 1
    col += 1

    newMat = np.ones((row, col))
    newMat *= -1

    newMat[0:row-1, 0:col-1] = A.copy()
    newMat[row-1,:] *= np.arange(0, col)
    newMat[:,col-1] *= np.arange(0, row)
    newMat[-1, -1] = 2
    return newMat

def deleteColByLabel(A, label):
    '''
    Delete the the column in A with specified label.
    The original matrix is not modified.

    Arguments:
        A {np.ndarray} -- matrix to be deleted.
        label {int} -- The label used in the matrix.
    Returns:
        [np.ndarray] -- A copy of the modified matrix.
    Raises:
        IndexError -- The label is not found.
    '''
    row, col = A.shape
    try:
        for colIdx in range(len(A[-1,:])):
            if A[-1,colIdx] == label:
                return np.delete(A, (colIdx), axis=1)
        print("___Label {} is not found in matrix\n{}___".format(label, A))
        raise IndexError
    except IndexError:
        print("IndexError. Index {}, matrix: {}".format(A,A))
        return
def getColIndexFromLabel(A, label):
    '''
    Find the index of the column with specific label in the matrix index system.

    Arguments:
        A {np.ndarray} -- matrix to be analyzed.
        label {int} -- The label used in the matrix.
    Returns:
        int -- The index of the column.
    Raises:
        IndexError -- The label is not found.
    '''
    for i in range(len(A[-1,:])):
        if A[-1,i] == label:
            return i
    print("___Label {} is not found in matrix\n{}___".format(label, A))
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
        # Start from the column c with the fewest "1"s.
        temp = A[0:-1, 0:-1]
        c = temp.sum(axis=0).argmin() # This is the index of the column.
        # Try each row that has "1".
        for r in range(len(A[:,c])):

            # Skip row that is not "1".
            if A[r,c] != 1:
                continue
            ansIndex = A[r,-1]
            B = A.copy()

            # Now iterate over row r and get col index that is "1".
            for j in range(len(A[r,:])):
                if A[r,j] == 1: # j now is the col index with "1".
                    # First delete the rows with ones in this col
                    # convert j (index in A) to label to (index in B)
                    label = A[-1,j]
                    idx = getColIndexFromLabel(B, label)
                    B = B[B[:,idx] != 1] # Use a mask to modify the matrix rows.

                    # Delete the column
                    B = deleteColByLabel(B, A[-1,j])

                # Skip col that is not "1":
                if A[r,j] != 1:
                    continue

            for sol in solveMatrix(B):
                yield [ansIndex] + sol

def makeMatrix(board, pents):
    """
    Accept the borad matrix and pentomino list and construct the matrix to be solved.

    Arguments:
        board {np-ndarray} -- The borad.
        pents {list} -- list of np-ndarray represent the possible pentominos.
    """

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

    raise NotImplementedError

def testSolveMatrix():
    '''
    Simple tests cases.
    First = [-5, -1, -3]
    Second = [-3, 0, -4]
    Besed on referenced code.
    '''
    # Given a 2D numpy array of 1 and 0, get its exact cover rows
    # row0 = np.array([0, 0, 1, 1, 0, 0])
    # row1 = np.array([1, 1, 0, 0, 0, 0])
    # row2 = np.array([0, 1, 0, 1, 0, 0])
    # row3 = np.array([0, 0, 1, 0, 0, 1])
    # row4 = np.array([1, 0, 0, 0, 0, 0])
    # row5 = np.array([0, 0, 0, 1, 1, 0])
    # MAT = np.array([row0, row1, row2, row3, row4, row5])

    # row0 = np.array([0, 0, 1, 0, 1, 1, 0])
    # row1 = np.array([1, 0, 0, 1, 0, 0, 1])
    # row2 = np.array([0, 1, 1, 0, 0, 1, 0])
    # row3 = np.array([1, 0, 0, 1, 0, 0, 0])
    # row4 = np.array([0, 1, 0, 0, 0, 0, 1])
    # row5 = np.array([0, 0, 0, 1, 1, 0, 1])
    # MAT = np.array([row0, row1, row2, row3, row4, row5])

    # row0 = np.array([1, 0, 0, 1, 0, 0, 1])
    # row1 = np.array([1, 0, 0, 1, 0, 0, 0])
    # row2 = np.array([0, 0, 0, 1, 1, 0, 1])
    # row3 = np.array([0, 0, 1, 0, 1, 1, 0])
    # row4 = np.array([0, 1, 1, 0, 0, 1, 1])
    # row5 = np.array([0, 1, 0, 0, 0, 0, 1])
    # MAT = np.array([row0, row1, row2, row3, row4, row5]) # Ans = [-1, -3, -5]

    row0 = np.array([1, 1, 0, 0, 0, 0])
    row1 = np.array([0, 0, 0, 0, 1, 1])
    row2 = np.array([0, 0, 0, 1, 1, 0])
    row3 = np.array([1, 1, 1, 0, 0, 0])
    row4 = np.array([0, 0, 1, 1, 0, 0])
    row5 = np.array([0, 0, 0, 1, 1, 0])
    row6 = np.array([1, 0, 1, 0, 1, 1])
    MAT = np.array([row0, row1, row2, row3, row4, row5, row6]) # Ans = [-0, -1, -4]


    MAT = addLabel(MAT)

    for i in solveMatrix(MAT):
        print(i)

def testSolveMatrixSpeed(number):
    row0 = np.array([0, 0, 1, 0, 1, 1, 0])
    row1 = np.array([1, 0, 0, 1, 0, 0, 1])
    row2 = np.array([0, 1, 1, 0, 0, 1, 0])
    row3 = np.array([1, 0, 0, 1, 0, 0, 0])
    row4 = np.array([0, 1, 0, 0, 0, 0, 1])
    row5 = np.array([0, 0, 0, 1, 1, 0, 1])
    MAT = np.array([row0, row1, row2, row3, row4, row5])

    row0 = np.array([1, 0, 0, 1, 0, 0, 1])
    row1 = np.array([1, 0, 0, 1, 0, 0, 0])
    row2 = np.array([0, 0, 0, 1, 1, 0, 1])
    row3 = np.array([0, 0, 1, 0, 1, 1, 0])
    row4 = np.array([0, 1, 1, 0, 0, 1, 1])
    row5 = np.array([0, 1, 0, 0, 0, 0, 1])
    row6 = np.array([1, 0, 0, 1, 0, 0, 0])
    MAT = np.array([row0, row1, row2, row3, row4, row5, row6]) # Ans = [6, 4, 2] [6, 4, 7]

    MAT = addLabel(MAT)

    for i in range(number):
        for j in solveMatrix(MAT):
            print(j)

if __name__ == "__main__":
    pass

testSolveMatrixSpeed(1)

A = np.genfromtxt("matrix.csv", delimiter=",")
# print(A[0,0:20])
A = A[1::,1::]
# print(A.shape)
# print(type(A))
# print(A[0,0:20])
np.random.seed(10)
np.random.shuffle(A)
A = A[:,0:60] # Cannot get over 62, 61x72 is 0.25s....

import time
t0 = time.time()
for i in solveMatrix(A):
    print(i)
    break
print("Total running time is {} s".format((time.time() - t0) / 1))
