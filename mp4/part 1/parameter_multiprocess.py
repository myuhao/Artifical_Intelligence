import multiprocessing
import time
import os

import numpy as np
from scipy import stats
import pygame
from pygame.locals import *
import pandas as pd
from matplotlib import pyplot as plt

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

def get_fname(Ne, C, gamma, avg):
    fname = "Ne-{:.2f} C-{:.2f} gamma-{:.2f} avg-{:.5f}.npy".format(Ne, C, gamma, avg)
    return fname

class Application:
    def __init__(self, parameters, snake_head_x=200, snake_head_y=200, food_x=80, food_y=80, **kwargs):
        self.Ne, self.C, self.gamma = parameters
        self.env = SnakeEnv(snake_head_x, snake_head_y, food_x, food_y)
        self.agent = Agent(self.env.get_actions(), self.Ne, self.C, self.gamma)
        self.train_eps = kwargs['train_eps']
        self.test_eps = kwargs['test_eps']
        self.check_converge = kwargs['check_converge']
        self.fname = kwargs['fname']
        self.process_name = self.fname
        self.fname = "{}_parameters.csv".format(self.fname)
        self.finished_run = False

    def train(self):
        print("{} - Sarting training:".format(self.process_name))
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

                if ct % 10000 == 0:
                    dt = time.time() - start_t_while
                    if dt > 45:
                        print(bcolors.FAIL + "{} - Running over time limit: {:.2f} s with {} iterations".format(self.process_name, dt, ct) + bcolors.ENDC)
                        return

            points = self.env.get_points()
            self.points_results.append(points)

            window = 1000
            if game % window == 0:
                print("{} - Training game {} - {}: ".format(self.process_name, game-window, game),
                      "Average: {:.2f}, Max: {}, Min: {}".format(
                        np.mean(self.points_results[-window:]),
                        np.max(self.points_results[-window:]),
                        np.min(self.points_results[-window:])))
                if self.check_converge:
                    if self.converge_calc(self.points_results, window):
                        self.train_eps = game
                        break

            self.env.reset()

        print(bcolors.OKGREEN + "{} - Training finshed with {} episodes, time {:.2f} s".format(self.process_name, self.train_eps, time.time() - start_t) + bcolors.ENDC)
        self.finished_run = True

    def test(self):
        print("{} - Starting testing:".format(self.process_name))
        self.agent.eval()
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
        print(bcolors.OKGREEN + "{} - Testing finshed with {} episodes, time {:.2f} s".format(self.process_name, self.test_eps, time.time() - start_t))
        print(bcolors.UNDERLINE + "Avg is {:.2f}".format(avg) + bcolors.ENDC + bcolors.ENDC)
        print("Max is {}".format(np.max(res)))
        print("Min is {}".format(np.min(res)))
        fname = "./points/{}".format(get_fname(self.Ne, self.C, self.gamma, avg))
        np.save(fname, np.concatenate([res, np.array([self.train_eps])]))
        try:
            with open(self.fname, 'a') as f:
                msg = "{},{},{},{},{},{},{},{},{},{}\n".format(self.Ne, self.C, self.gamma, avg, np.max(res), np.min(res), self.train_eps, self.test_eps, get_fname(self.Ne, self.C, self.gamma, avg), "fixed")
                f.write(msg)
        except FileNotFoundError:
            with open(self.fname, 'w') as f:
                msg = "Ne,C,gamma,avg,max,min,train_eps,test_eps,npy_fname\n"
                f.write(msg)

    def excute(self):
        self.train()
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

def run_process(parameters, name, train_eps=50000):
    total = len(parameters)
    ct = 0
    for p in parameters:
        app = Application(p, train_eps=train_eps, test_eps=1000, check_converge=True, fname=name)
        app.excute()
        ct += 1
        n, c, g = p
        print("{} - Parameters used: Ne-{}, alpha-{}, gamma-{}".format(name, n, c, g))
        print(bcolors.OKGREEN + "---------------------------------{}: {}/{}-------------------------------------".format(name, ct, total) + bcolors.ENDC)
        print()

def split_parameters(Ne, C, gamma):
    para = []
    para_0 = []
    para_1 = []
    para_2 = []
    para_3 = []

    for n in Ne:
        for c in C:
            for g in gamma:
                parameter = (n, c, g)
                para.append(parameter)

    index = np.arange(len(para))
    np.random.shuffle(index)
    for i in range(len(index)):
        if i % 4 == 0:
            para_0.append(para[index[i]])
        elif i % 4 == 1:
            para_1.append(para[index[i]])
        elif i % 4 == 2:
            para_2.append(para[index[i]])
        else:
            para_3.append(para[index[i]])
    return para_0, para_1, para_2, para_3

def main():

    Ne_list = np.array([30])
    C_list = np.arange(0.1, 1, 0.1)
    gamma_list = np.array([0.5, 0.9])

    para_0, para_1, para_2, para_3 = split_parameters(Ne_list, C_list, gamma_list)

    arg0 = (para_0, 'process_0')
    arg1 = (para_1, 'process_1')
    arg2 = (para_2, 'process_2')
    arg3 = (para_3, 'process_3')

    p0 = multiprocessing.Process(target=run_process, args=arg0)
    p1 = multiprocessing.Process(target=run_process, args=arg1)
    p2 = multiprocessing.Process(target=run_process, args=arg2)
    p3 = multiprocessing.Process(target=run_process, args=arg3)

    p0.start()
    print('p0 starts')
    p1.start()
    print('p1 starts')
    p2.start()
    print('p2 starts')
    p3.start()
    print('p3 starts')

    p0.join()
    p1.join()
    p2.join()
    p3.join()

if __name__ == '__main__':
    data = np.load('./points/Ne-41.111111111111114 C-42.0 gamma-0.50 avg-24.47400.npy')
    l = len(data) - 1
    plt.plot(np.arange(l), data[:-1], 'k.')
    plt.show()
