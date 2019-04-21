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

def get_fname(Ne, C, gamma, avg):
	fname = "Ne-{} C-{} gamma-{:.2f} avg-{:.5f}.npy".format(Ne, C, gamma, avg)
	return fname

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
				if self.check_converge:
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

		self.res = np.array(points_results)
		res = self.res
		avg = np.mean(res)
		print(bcolors.OKGREEN + "Testing finshed with {} episodes, time {:.2f} s".format(self.test_eps, time.time() - start_t))
		print(bcolors.UNDERLINE + "Avg is {:.2f}".format(avg) + bcolors.ENDC + bcolors.ENDC)
		print("Max is {}".format(np.max(res)))
		print("Min is {}".format(np.min(res)))
		with open("parameters.csv", 'a') as f:
			msg = "{},{},{},{},{},{},{},{}\n".format(self.Ne, self.C, self.gamma, avg, np.max(res), np.min(res), self.train_eps, self.test_eps)
			f.write(msg)

		fname = "./points/{}".format(get_fname(self.Ne, self.C, self.gamma, avg))
		np.save(fname, np.concatenate([res, np.array([self.train_eps])]))


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
				app = Application(parameters, train_eps=50000, test_eps=1000, check_converge=True)
				app.excute()
				ct += 1
				print("{}/{} finished".format(ct, total))
				print("Parameters used: Ne-{}, C-{}, gamma-{}".format(n, c, g))
				print("----------------------------------------------------------------------")
				print()

def read_top_results(top):
	df = pd.read_csv('parameters.csv')
	print(df.shape)
	df0 = pd.read_csv('process_0_parameters.csv')
	df1 = pd.read_csv('process_1_parameters.csv')
	df2 = pd.read_csv('process_2_parameters.csv')
	df3 = pd.read_csv('process_3_parameters.csv')
	frames = [df, df0, df1, df2, df3]
	df = pd.concat(frames)
	df = df.drop(['npy_fname'], axis=1)
	top_df = df[df['train_eps'] > 10000].nlargest(top, 'avg')
	# top_df = top_df.drop(['npy_fname'], axis=1)
	print(top_df.sort_values(by=['avg'], ascending=False))

def run_exaust():
	parameters_list = [(50, 10, 0.3), (30, 40, 0.6), (70, 70, 0.1), (30, 70, 0.1), (50, 50, 0.1), (30, 30, 0.7)]
	ct = 0
	total = len(parameters_list)
	for parameters in parameters_list:
		# print(bcolors.WARNING + "Parameters used: Ne-{}, C-{}, gamma-{}".format(n, c, g) + bcolors.ENDC)
		app = Application(parameters, train_eps=100000, test_eps=1000, check_converge=False)
		app.excute()
		ct += 1
		print("{}/{} finished".format(ct, total))
		print("----------------------------------------------------------------------")
		print()

def test_gamma():
	gamma = np.linspace(0.15, 0.95, 9)
	Ne = 30
	C = 30
	avg = []
	for g in gamma:
		parameters = (Ne, C, g)
		app = Application(parameters, train_eps=50000, test_eps=1000, check_converge=True)
		app.excute()

def plot_gamma():
	df = pd.read_csv('parameters.csv')
	df = df[df['train_eps'] > 10000]
	df = df[df['Ne'] == 30]
	df = df[df['C'] == 30]
	gamma = df['gamma']
	avg = df['avg']
	print(df)
	plt.scatter(gamma, avg)
	plt.show()

def test_Ne():
	Ne = np.linspace(30, 50, 10)
	C = 40
	gamma = 0.5
	for n in Ne:
		parameters = (n, C, gamma)
		app = Application(parameters, train_eps=50000, test_eps=1000, check_converge=True)
		app.excute()

def test_C():
	Ne = 40
	C = np.linspace(10,50,10)
	gamma = 0.3
	for c in C:
		parameters = (Ne, c, gamma)
		app = Application(parameters, train_eps=50000, test_eps=1000, check_converge=True)
		app.excute()


def long_test():
	Ne = np.linspace(30, 50, 10)
	C = np.linspace(10,50,6)
	gamma = [0.3, 0.5, 0.7, 0.9]
	total = len(Ne) * len(C) * len(gamma)
	ct = 0
	for n in Ne:
		for c in C:
			for g in gamma:
				# Ne, C, gamma
				parameters = (n, c, g)
				app = Application(parameters, train_eps=50000, test_eps=1000, check_converge=True)
				app.excute()
				ct += 1
				print("{}/{} finished".format(ct, total))
				print("Parameters used: Ne-{}, C-{}, gamma-{}".format(n, c, g))
				print("----------------------------------------------------------------------")
				print()

def visualize():
	df = pd.read_csv('parameters.csv')
	df0 = pd.read_csv('process_0_parameters.csv')
	df1 = pd.read_csv('process_1_parameters.csv')
	df2 = pd.read_csv('process_2_parameters.csv')
	df3 = pd.read_csv('process_3_parameters.csv')
	frames = [df, df0, df1, df2, df3]
	df = pd.concat(frames)
	df = df.drop(['npy_fname'], axis=1)
	df = df[df['train_eps'] > 10000]
	Ne = df['Ne']
	C = df['C']
	gamma = df['gamma']
	avg = df['avg']

	def plot_subplot(ax, name):
		x = df[name]
		ax.scatter(x, avg)
		ax.set(title=name)

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	plot_subplot(ax1, 'Ne')
	plot_subplot(ax2, 'C')
	plot_subplot(ax3, 'gamma')
	fig.tight_layout()

	# C_idx = df['C'] == 30
	# gamma = gamma[C_idx]
	# avg = avg[C_idx]

	# Ne_idx = df['Ne'] == 50
	# gamma = gamma[Ne_idx]
	# avg = avg[Ne_idx]
	# print(avg)
	# plt.scatter(gamma, avg)
	plt.show()


if __name__ == "__main__":
	# # Ne, C, gamma
	# parameters = (50, 10, 0.3)
	# app = Application(parameters, train_eps=100000, test_eps=1000, check_converge=False)
	# app.excute()

	# main()
	# run_exaust()
	read_top_results(20)
	visualize()
	# plot_gamma()
	# long_test()


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
