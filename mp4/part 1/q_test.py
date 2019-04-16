import numpy as np
import pygame
from pygame.locals import *
import time

import utils
from agent import Agent
from snake import SnakeEnv


class Application:
    def __init__(self, case, fname):
        snake_head_x, snake_head_y, food_x, food_y, Ne, C, gamma = case
        self.env = SnakeEnv(snake_head_x, snake_head_y, food_x, food_y)
        self.agent = Agent(self.env.get_actions(), Ne, C, gamma)
        self.train_eps = 10000
        self.fname = fname

    def train(self):
        print("Start training:")
        self.agent.train()
        first_eat = True
        start = time.time()

        for game in range(self.train_eps):
            state = self.env.get_state()
            dead = False
            action = self.agent.act(state, 0, dead)
            while not dead:
                start, points, dead = self.env.step(action)

                if first_eat and points == 1:
                    self.agent.save_model(self.fname)
                    first_eat = False
                    print("Debugging model saved {}".format(self.fname))
                    return

                action = self.agent.act(state, points, dead)

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
        self.cases = [(200, 200, 80, 80, 40, 40, 0.7),
                      (200, 200, 80, 80, 20, 60, 0.5),
                      (80, 80, 200, 200, 40, 40, 0.7)]
        self.my_checkpoints = ["./test/my_checkpoint1.npy",
                               "./test/my_checkpoint2.npy",
                               "./test/my_checkpoint3.npy"]
        self.checkpoints = [self.checkpoint1,
                            self.checkpoint2,
                            self.checkpoint3]

    def diff(self, myAns, checkpoint):
        diffIndex = np.nonzero(myAns != checkpoint)
        print(myAns[diffIndex])
        print(checkpoint1[diffIndex])


    def test(self, case_num):
        case = self.cases[case_num-1]
        fname = self.my_checkpoints[case_num-1]
        app = Application(case, fname)
        app.train()
        # self.diff(case_num)



if __name__ == "__main__":
    myAns = utils.load("./checkpoint.npy")
    checkpoint = utils.load("./test/checkpoint1.npy")
    diffIndex = np.nonzero(myAns != checkpoint)
    print(myAns[diffIndex])
    print(checkpoint[diffIndex])
