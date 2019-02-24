import pandas as pd
import numpy as np


pentominoes = [
    np.array(p) for p in [

        # F
        [[0,1,1],
         [1,1,0],
         [0,1,0]],

        # I
        [[1,1,1,1,1]],

        # L
        [[1,1,1,1],
         [0,0,0,1]],

        # N
        [[1,1,0,0],
         [0,1,1,1]],

        # P
        [[1,1,1],
         [0,1,1]],

        # T
        [[1,1,1],
         [0,1,0],
         [0,1,0]],

        # U
        [[1,0,1],
         [1,1,1]],

        # V
        [[1,0,0],
         [1,0,0],
         [1,1,1]],

        # W
        [[1,0,0],
         [1,1,0],
         [0,1,1]],

        # X
        [[0,1,0],
         [1,1,1],
         [0,1,0]],

        # Y
        [[0,0,1,0],
         [1,1,1,1]],

        # Z
        [[1,1,0],
         [0,1,0],
         [0,1,1]]
    ]
]
# pentominoes is a list of all possible pentomino.

def all_orientations(A, i):
    """Generate all distinct orientations of the pentominoes,
    including rotations and reflections."""

    # Fixing the orientation of the first (F) pentomino eliminates
    # redundant solutions resulting from rotations or reflections."""
    if i == 0:
        yield A
        return

    seen = set()
    # Apply transpose, left/right flip, and up/down flip in all combinations
    # to generate all possible orientiations of a pentomino.
    for A in (A, A.T):
        for A in (A, np.fliplr(A)):
            for A in (A, np.flipud(A)):
                s = str(A)
                if not s in seen:
                    yield A
                    seen.add(s)


def all_positions(A, i):
    """ Find all positions to place the pentominoes. """
    for A in all_orientations(A, i):
        rows, cols = A.shape
        for i in range(9 - rows):
            for j in range(9 - cols):
                M = np.zeros((8, 8), dtype='int')
                M[i:i+rows, j:j+cols] = A
                if M[0,0] == M[0,7] == M[7,0] == M[7,7] == 0:
                    yield np.delete(M.reshape(64), [0, 7, 56, 63])

def exact_cover(A):
    # If matrix has no columns, terminate successfully.
    if A.shape[1] == 0:
        yield []
    else:
        # Choose a column c with the fewest 1s.
        c = A.sum(axis=0).argmin()
        # For each row r such that A[r,c] = 1,
        for r in A.index[A[c] == 1]: # Index of the 1 row.
            B = A

            # For each column j such that A[r,j] = 1,
            for j in A.columns[A.loc[r] == 1]: # Index of columns in row r that is "1".

                # Delete each row i such that A[i,j] = 1
                B = B[B[j] == 0]

                # then delete column j.
                del B[j]



            for partial_solution in exact_cover(B):
                # Include r in the partial solution.
                yield [r] + partial_solution

rows = []
for i, P in enumerate(pentominoes):
    for A in all_positions(P, i):
        A = np.append(A, np.zeros(12, dtype='int'))
        A[60+i] = 1
        rows.append(list(A))

A = pd.DataFrame(rows)
A.to_csv('matrix.csv')

# import time
# t0 = time.time()

# for i in exact_cover(A):
#     print(i)
#     # break

# print("Running time is {0:.3f}s.".format((time.time()-t0)/60))

# covers = np.array(list(exact_cover(A)), dtype='int')
# np.savetxt('exact-covers.csv', covers, delimiter=',', fmt='%d')


# if __name__ == "__main__":

#     # Given a 2D numpy array of 1 and 0, get its exact cover rows
#     # row0 = np.array([0, 0, 1, 1, 0, 0])
#     # row1 = np.array([1, 1, 0, 0, 0, 0])
#     # row2 = np.array([0, 1, 0, 1, 0, 0])
#     # row3 = np.array([0, 0, 1, 0, 0, 1])
#     # row4 = np.array([1, 0, 0, 0, 0, 0])
#     # row5 = np.array([0, 0, 0, 1, 1, 0])
#     # MAT = np.array([row0, row1, row2, row3, row4, row5])

#     row0 = np.array([0, 0, 1, 0, 1, 1, 0])
#     row1 = np.array([1, 0, 0, 1, 0, 0, 1])
#     row2 = np.array([0, 1, 1, 0, 0, 1, 0])
#     row3 = np.array([1, 0, 0, 1, 0, 0, 0])
#     row4 = np.array([0, 1, 0, 0, 0, 0, 1])
#     row5 = np.array([0, 0, 0, 1, 1, 0, 1])
#     MAT = np.array([row0, row1, row2, row3, row4, row5])
#     # MAT = addIndex(MAT)

#     MAT = pd.DataFrame(MAT)

#     # for j in range(1):
#     #     for i in exact_cover(MAT):
#     #         print(i)


