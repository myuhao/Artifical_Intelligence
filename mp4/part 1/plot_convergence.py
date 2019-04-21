import numpy as np
import pygame
from pygame.locals import *
import time
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import utils
from agent import Agent
from snake import SnakeEnv

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Application:
    def __init__(self, parameters, snake_head_x=200, snake_head_y=200, food_x=80, food_y=80, **kwargs):
        self.Ne, self.C, self.gamma = parameters
        self.env = SnakeEnv(snake_head_x, snake_head_y, food_x, food_y)
        self.agent = Agent(self.env.get_actions(), self.Ne, self.C, self.gamma)
        self.train_eps = kwargs['train_eps']
        self.test_eps = kwargs['test_eps']
        self.check_converge = kwargs['check_converge']
        self.modle_fname = "temp.npy"
        self.finished_run = False
        self.window = 1000
        self.avgPt = []
        self.tick = []

    def train(self):
        print("Sarting training:")
        self.agent.train()
        self.points_results = []

        start_t = time.time()

        for game in range(1, self.train_eps + 1):
            state = self.env.get_state()
            dead = False
            action = self.agent.act(state, 0, dead)
            ct = 0
            start_t_while = time.time()
            while not dead:
                state, points, dead = self.env.step(action)
                action = self.agent.act(state, points, dead)
                ct += 1
                # if ct >= 50000:
                #   print(bcolors.FAIL + "Running over time limit: {}".format(ct) + bcolors.ENDC)
                #   return
                if ct % 10000 == 0:
                    dt = time.time() - start_t_while
                    if dt > 45:
                        print(bcolors.FAIL + "Running over time limit: {:.2f} s with {} iterations".format(dt, ct) + bcolors.ENDC)
                        return

            points = self.env.get_points()
            self.points_results.append(points)

            if game % self.window == 0:
                self.avgPt.append(np.mean(self.points_results[-self.window:]))
                self.tick.append(game)
                print("Training game {} - {}: ".format(game-self.window, game),
                      "Average: {:.2f}, Max: {}, Min: {}".format(
                        np.mean(self.points_results[-self.window:]),
                        np.max(self.points_results[-self.window:]),
                        np.min(self.points_results[-self.window:])))
                if self.check_converge:
                    if self.converge_calc(self.points_results, window):
                        self.train_eps = game
                        break

            self.env.reset()

        print(bcolors.OKGREEN + "Training finshed with {} episodes, time {:.2f} s".format(self.train_eps, time.time() - start_t) + bcolors.ENDC)
        self.finished_run = True
        save_data = np.array([self.tick, self.avgPt])
        np.save("temp.npy", save_data)

    def test(self):
        print("Starting testing:")
        self.agent.eval()
        # self.agent.load_model(self.modle_fname)
        points_results = []
        start_t = time.time()

        for game in range(1, self.test_eps + 1):
            state = self.env.get_state()
            dead = False
            action = self.agent.act(state, 0, dead)

            while not dead:
                state, points, dead = self.env.step(action)
                action = self.agent.act(state, points, dead)

            points = self.env.get_points()
            points_results.append(points)
            self.env.reset()

        self.res = np.array(points_results)
        res = self.res
        avg = np.mean(res)
        print(bcolors.OKGREEN + "Testing finshed with {} episodes, time {:.2f} s".format(self.test_eps, time.time() - start_t))
        print(bcolors.UNDERLINE + "Avg is {:.2f}".format(avg) + bcolors.ENDC + bcolors.ENDC)

    def excute(self):
        self.train()
        self.plot()
        if self.finished_run:
            self.test()

    def converge_calc(self, values, window):
        init_window = 5
        if len(values) <= init_window * window:
            return False

        y = np.array(values[-window * init_window:])
        x = np.linspace(np.min(y), np.max(y), len(y))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        msg = "Slope is {:.4f}".format(slope)
        print(bcolors.OKBLUE + msg + bcolors.ENDC) if slope > 0 else print(bcolors.WARNING + msg + bcolors.ENDC)

        return np.isclose(slope, 0, rtol=0, atol=0.001)

    def plot(self):
        try:
            data = np.load('temp.npy')
            self.tick = data[0]
            self.avgPt = data[1]
            print("Load data!")
        except:
            print("Not found file")
            pass
        plt.plot(self.tick, self.avgPt, 'k.')
        plt.xlabel("Game Number")
        plt.ylabel("Average Score")
        plt.title("The Average Scores During Training")
        plt.savefig("Ne-30_alpha-0.2_gamma-0.5.png")


def main():
    parameters = (30, 34, 0.5)
    app = Application(parameters, train_eps=50000, test_eps=1000, check_converge=False)
    app.excute()

main()

"""
Ne30 C30 gamma0.5 - 24.744
Ne30 alpha0.5 gamma0.5 - 28.70
[description]
"""
