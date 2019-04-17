import numpy as np
import pygame
from pygame.locals import *
import time
import pandas as pd
from scipy import stats

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
		self.modle_fname = "temp.npy"
		self.finished_run = False

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
				# 	print(bcolors.FAIL + "Running over time limit: {}".format(ct) + bcolors.ENDC)
				# 	return
				if ct % 10000 == 0:
					dt = time.time() - start_t_while
					if dt > 45:
						print(bcolors.FAIL + "Running over time limit: {:.2f} s with {} iterations".format(dt, ct) + bcolors.ENDC)
						return

			points = self.env.get_points()
			self.points_results.append(points)

			window = 1000
			if game % window == 0:
				print("Training game {} - {}: ".format(game-window, game),
					  "Average: {:.2f}, Max: {}, Min: {}".format(
					  	np.mean(self.points_results[-window:]),
					  	np.max(self.points_results[-window:]),
					  	np.min(self.points_results[-window:])))
				if self.converge_calc(self.points_results, window):
					self.train_eps = game
					break

			self.env.reset()

		print(bcolors.OKGREEN + "Training finshed with {} episodes, time {:.2f} s".format(self.train_eps, time.time() - start_t) + bcolors.ENDC)
		# self.agent.save_modle(self.modle_fname)
		self.finished_run = True

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

		res = np.array(points_results)
		avg = np.mean(res)
		print(bcolors.OKGREEN + "Testing finshed with {} episodes, time {:.2f} s".format(self.test_eps, time.time() - start_t))
		print(bcolors.UNDERLINE + "Avg is {:.2f}".format(avg) + bcolors.ENDC)
		print("Max is {}".format(np.max(res)))
		print("Min is {}".format(np.min(res)))
		with open("parameters.csv", 'a') as f:
			msg = "{},{},{},{},{},{},{},{}\n".format(self.Ne, self.C, self.gamma, avg, np.max(res), np.min(res), self.train_eps, self.test_eps)
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
		print("Slope is {:.4f}".format(slope))

		return np.isclose(slope, 0, rtol=0, atol=0.001)

def main():
	Ne = np.linspace(10, 90, 5)
	C = np.linspace(10, 90, 5)
	gamma = np.linspace(0.1, 0.9, 5)
	total = len(Ne) * len(C) * len(gamma)
	ct = 0
	for n in Ne:
		for c in C:
			for g in gamma:
				# Ne, C, gamma
				parameters = (n, c, g)
				app = Application(parameters, train_eps=50000, test_eps=1000)
				app.excute()
				ct += 1
				print("{}/{} finished".format(ct, total))
				print("Parameters used: Ne-{}, C-{}, gamma-{}".format(n, c, g))
				print("----------------------------------------------------------------------")
				print()

def read_top_results(top):
	df = pd.read_csv('parameters.csv')
	print(df.nlargest(top, 'avg'))

if __name__ == "__main__":
	# # Ne, C, gamma
	# parameters = (30, 40, 0.6)
	# app = Application(parameters, train_eps=50000, test_eps=1000)
	# app.excute()

	main()
	read_top_results(20)


'''
Tried parameters:
train_eps = 10000:
Ne:    [10, 30, 50, 70, 90, 35, 40, 45, 50]
C:     [10, 30, 50, 70, 90, 35, 40, 45, 50]
gamma: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

train_eps = 50000:
Ne:    [30, 40]
C:     [30, 40, 50]
gamma: [0.6, 0.7, 0.8]

train_eps = 50000 w/ conergence tests:
Ne:    [10, 30, 50, 70, 90]
C:     [10, 30, 50, 70, 90]
gamma: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
'''
