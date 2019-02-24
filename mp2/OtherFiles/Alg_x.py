import numpy as np
import pandas as pd

def solve(X, Y, solution=[]):
    '''
    Solve the exact cover problem using algorithm X.
    ref = "https://www.cs.mcgill.ca/~aassaf9/python/algorithm_x.html"

    Arguments:
        X {dict} -- Board coordinates (in 1D) map to tile identity (as string).
        Y {dict} -- Tile identity (as string) map to a list that contain all the borad coordinates (in 1D) it coocupies.

    Keyword Arguments:
        solution {list} -- [description] (default: {[]})

    Yields:
        [list] -- List of tile in particular orentation at certain location to be used.
    '''
    if not X:
        yield list(solution)
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution.append(r)
            cols = select(X, Y, r)
            for s in solve(X, Y, solution):
                yield s
            deselect(X, Y, r, cols)
            solution.pop()

def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols

def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)

X = {
    1: {'A', 'B'},
    2: {'E', 'F'},
    3: {'D', 'E'},
    4: {'A', 'B', 'C'},
    5: {'C', 'D'},
    6: {'D', 'E'},
    7: {'A', 'C', 'E', 'F'}}
Y = {
    'A': [1, 4, 7],
    'B': [1, 4],
    'C': [4, 5, 7],
    'D': [3, 5, 6],
    'E': [2, 3, 6, 7],
    'F': [2, 7]}

# Given a 2D numpy array of 1 and 0, get its exact cover rows

row0 = np.array([0, 0, 1, 0, 1, 1, 0])
row1 = np.array([1, 0, 0, 1, 0, 0, 1])
row2 = np.array([0, 1, 1, 0, 0, 1, 0])
row3 = np.array([1, 0, 0, 1, 0, 0, 0])
row4 = np.array([0, 1, 0, 0, 0, 0, 1])
row5 = np.array([0, 0, 0, 1, 1, 0, 1])
MAT = np.array([row0, row1, row2, row3, row4, row5])


# def twoD2oneD(x, y):
#     return y + 7*x
# def oneD2twoD(n):
#     return n % 7, n // 7

# flatMat = np.arange(42)
# X = {}
# Y = {}

# # Y is the dict that map which pos in the mat is used.
# for i in range(len(MAT)):
#     coverX = []
#     for j in range(len(MAT[i])):
#         if MAT[i][j] == 0:
#             continue
#         coverX.append(twoD2oneD(i, j))
#     Y[i] = coverX
# # print(Y)

# # X is the dict map each coords to the key.
# for i in range(len(MAT)):
#     for j in range(len(MAT[i])):
#         if MAT[i][j] == 0:
#             X[twoD2oneD(i,j)] = {}
#         else:
#             X[twoD2oneD(i,j)] = {i}
# # print(X)

# X = {j: set() for j in X}
# for i in Y:
#     for j in Y[i]:
#         X[j].add(i)

# print(X)

# for s in solve(X, Y):
#     print(s)


col = {}
row = {}

for i in range(len(MAT)):
    row[i] = True

for i in range(len(MAT[0])):
    col[i] = True

def solveMat(matrix, col, row):
    print("len(col) is: {}".format(len(col)))
    # Base case.
    if len(col) == 0 and len(row) == 0:
        print("Solution found")
        return sol
    if len(col) == 0 or len(row) == 0:
        print("No solution in this case.")
        return sol

    NUM_ROW, NUM_COL = matrix.shape

    # Begin recursive

    # Iterate over all columns
    for c in range(1):
        # this is to select column c.
        for r in range(NUM_ROW):
            # Try to locate the the start.
            if matrix[r,c] == 1:
                row[r] = False
                for newCIndex in range(NUM_COL):
                    if matrix[r][newCIndex] == 1:
                        col[newCIndex] = False
                        for newRIndex in range(NUM_ROW):
                            if matrix[newRIndex][newCIndex] == 1:
                                row[newRIndex] = False
                # End of alg steps 1-3.
                # print(matrix)
                # print("Col is {}, row is {}".format(col, row))
                # return
                # Recurse down to the samller matrix.
                print("This is row {}".format(r))
                sol = []
                sol.append(r)
                remainCol = []
                remainRow = []
                for k in col:
                    if col[k] == True:
                        remainCol.append(k)
                for k in row:
                    if row[k] == True:
                        remainRow.append(k)
                if len(remainCol) == 0 or len(remainRow) == 0:
                    print("-----Failed-----")
                    return sol
                newMat = []
                for i in remainRow:
                    newR = []
                    for j in remainCol:
                        newR.append(matrix[i][j])
                    newMat.append(np.array(newR))
                newMat = np.array(newMat)
                print("Reduced matrix is: \n{}\n".format(newMat))
                newCol = {}
                newRow = {}
                for i in range(len(newMat)):
                    newRow[i] = True
                for i in range(len(newMat[0])):
                    newCol[i] = True
                sol.append(solveMat(newMat, newCol, newRow))
                return sol
    print("------------Reach end of the loop---------------")
    return sol



def solveMat(matrix):
    col = {}
    row = {}
    for i in range(len(matrix)):
        row[i] = True
    for i in range(len(matrix[0])):
        col[i] = True

    print("len(col) is: {}".format(len(col)))
    # Base case.
    if len(col) == 0 and len(row) == 0:
        print("Solution found")
        return sol
    if len(col) == 0 or len(row) == 0:
        print("No solution in this case.")
        return sol

    NUM_ROW, NUM_COL = matrix.shape

    # Begin recursive

    # Iterate over all columns
    for c in range(1):
        # this is to select column c.
        for r in range(NUM_ROW):
            # Try to locate the the start.
            if matrix[r,c] == 1:
                row[r] = False
                for newCIndex in range(NUM_COL):
                    if matrix[r][newCIndex] == 1:
                        col[newCIndex] = False
                        for newRIndex in range(NUM_ROW):
                            if matrix[newRIndex][newCIndex] == 1:
                                row[newRIndex] = False
                # End of alg steps 1-3.
                # print(matrix)
                # print("Col is {}, row is {}".format(col, row))
                # return
                # Recurse down to the samller matrix.
                print("This is row {}".format(r))
                sol = []
                sol.append(r)
                remainCol = []
                remainRow = []
                for k in col:
                    if col[k] == True:
                        remainCol.append(k)
                for k in row:
                    if row[k] == True:
                        remainRow.append(k)
                if len(remainCol) == 0 or len(remainRow) == 0:
                    print("-----Failed-----")
                    return r
                newMat = []
                for i in remainRow:
                    newR = []
                    for j in remainCol:
                        newR.append(matrix[i][j])
                    newMat.append(np.array(newR))
                newMat = np.array(newMat)
                print("Reduced matrix is: \n{}\n".format(newMat))

                sol.append(solveMat(newMat))
    print("------------Reach end of the loop---------------")
    return sol


a = solveMat(MAT)
print(a)

def exact_cover(A):
    # If matrix has no columns, terminate successfully.
    if A.shape[1] == 0:
        yield []
    else:
        # Choose a column c with the fewest 1s.
        c = A.sum(axis=0).argmin()

        # For each row r such that A[r,c] = 1,
        for r in A.index[A[c] == 1]:

            B = A

            # For each column j such that A[r,j] = 1,
            for j in A.columns[A.loc[r] == 1]:

                # Delete each row i such that A[i,j] = 1
                B = B[B[j] == 0]

                # then delete column j.
                del B[j]

            for partial_solution in exact_cover(B):
                # Include r in the partial solution.
                yield [r] + partial_solution


MAT = pd.DataFrame(MAT)
print(exact_cover(MAT))
for i in exact_cover(MAT):
    print(i)
