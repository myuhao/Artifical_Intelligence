import utils
import numpy as np

class Test:
    '''
    [checkpoint1.npy] snake_head_x=200, snake_head_y=200, food_x=80,  food_y=80,  Ne=40, C=40, gamma=0.7
    [checkpoint2.npy] snake_head_x=200, snake_head_y=200, food_x=80,  food_y=80,  Ne=20, C=60, gamma=0.5
    [checkpoint3.npy] snake_head_x=80,  snake_head_y=80,  food_x=200, food_y=200, Ne=40, C=40, gamma=0.7
    '''
    def __init__(self):
        self.myAns = utils.load("./checkpoint.npy")
        self.checkpoint1 = utils.load("./test/checkpoint1.npy")
        self.checkpoint2 = utils.load("./test/checkpoint2.npy")
        self.checkpoint3 = utils.load("./test/checkpoint3.npy")

    def diff(self):
        diffIndex = np.nonzero(self.myAns != self.checkpoint1)
        print(self.myAns[diffIndex])


if __name__ == "__main__":
    test = Test()
    test.diff()
