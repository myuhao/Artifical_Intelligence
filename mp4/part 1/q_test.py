import numpy as np
import pygame
from pygame.locals import *
import time

import utils
from agent import Agent
from snake import SnakeEnv


class Application:
    '''
    Copy codes from snake_main.py.
    '''
    def __init__(self, fname, case, **kwargs):
        snake_head_x, snake_head_y, food_x, food_y, Ne, C, gamma = case
        self.env = SnakeEnv(snake_head_x, snake_head_y, food_x, food_y)
        self.agent = Agent(self.env.get_actions(), Ne, C, gamma)
        self.train_eps = kwargs['train_eps']
        self.fname = fname

    def train(self):
        print("Start training")
        self.agent.train()
        first_eat = True
        start = time.time()

        for game in range(1, self.train_eps + 1):
            state = self.env.get_state()
            dead = False
            action = self.agent.act(state, 0, dead)
            while not dead:
                state, points, dead = self.env.step(action)

                if first_eat and points == 1:
                    self.agent.save_model(self.fname)
                    first_eat = False
                    print("Game {} finshed".format(game))
                    print("Debugging model saved as {}".format(self.fname))
                    return

                action = self.agent.act(state, points, dead)

            points = self.env.get_points()
            self.env.reset()
            if game % 100 == 0:
                print("Game {} - {} finished".format(game, game+100))

        print("Finished Training")

    def excute(self):
        self.train()

class Test:
    '''
    [checkpoint1.npy] snake_head_x=200, snake_head_y=200, food_x=80,  food_y=80,  Ne=40, C=40, gamma=0.7
    [checkpoint2.npy] snake_head_x=200, snake_head_y=200, food_x=80,  food_y=80,  Ne=20, C=60, gamma=0.5
    [checkpoint3.npy] snake_head_x=80,  snake_head_y=80,  food_x=200, food_y=200, Ne=40, C=40, gamma=0.7
    '''
    def __init__(self):
        self.cases = [(200, 200, 80, 80, 40, 40, 0.7),
                      (200, 200, 80, 80, 20, 60, 0.5),
                      (80, 80, 200, 200, 40, 40, 0.7)]
        self.my_checkpoints = ["./test/my_checkpoint1.npy",
                               "./test/my_checkpoint2.npy",
                               "./test/my_checkpoint3.npy"]
        self.checkpoints = ["./test/checkpoint1.npy",
                            "./test/checkpoint2.npy",
                            "./test/checkpoint3.npy"]

    def diff(self, case_num):

        myAns = utils.load(self.my_checkpoints[case_num-1])
        checkpoint = utils.load(self.checkpoints[case_num-1])
        diffIndex = np.nonzero(myAns != checkpoint)
        # diffIndex = np.nonzero(np.invert(np.isclose(myAns, checkpoint, rtol=1e-05, atol=1e-08)))
        diffIndex = np.nonzero(myAns != checkpoint)
        diffState = np.array(diffIndex)

        print(diffState)
        print("My answer gives: {}".format(myAns[diffIndex]))
        print("TA answer gives: {}".format(checkpoint[diffIndex]))

    def test(self, case_num):
        case = self.cases[case_num-1]
        fname = self.my_checkpoints[case_num-1]
        app = Application(fname, case, train_eps=1000)
        app.train()
        self.diff(case_num)

if __name__ == "__main__":
    t = Test()
    for i in range(1, 4):
        print("test case {}".format(i))
        t.test(i)
        print()
