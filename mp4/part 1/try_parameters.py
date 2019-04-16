import numpy as np
import pygame
from pygame.locals import *
import time

import utils
from agent import Agent
from snake import SnakeEnv


class Application:
	def __init__(self, parameters, snake_head_x=200, snake_head_y=200, food_x=80, food_y=80, **kwargs):
		self.Ne, self.C, self.gamma = parameters
		self.env = SnakeEnv(snake_head_x, snake_head_y, food_x, food_y)
		self.agent = Agent(self.env.get_actions(), self.Ne, self.C, self.gamma)
		self.train_eps = kwargs['train_eps']
		self.test_eps = kwargs['test_eps']
		self.modle_fname = "temp.npy"

	def train(self):
		print("Sarting training:")
		self.agent.train()
		self.points_results = []

		start_t = time.time()

		for game in range(1, self.train_eps + 1):
			state = self.env.get_state()
			dead = False
			action = self.agent.act(state, 0, dead)
			
			while not dead:
				state, points, dead = self.env.step(action)
				action = self.agent.act(state, points, dead)

			points = self.env.get_points()
			self.points_results.append(points)

			if game % 100 == 0:
				print("Training game {} - {}".format(game, game+100))

			self.env.reset()

		print("Training finshed with {} episodes, time {:.2f} s".format(self.train_eps, time.time() - start_t))
		# self.agent.save_modle(self.modle_fname)

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
			self.points_results.append(points)
			self.env.reset()

		res = np.array(self.points_results)
		avg = np.mean(res)
		print("Testing finshed with {} episodes, time {:.2f} s".format(self.test_eps, time.time() - start_t))
		print("Avg is {:.2f}".format(avg))
		print("Max is {}".format(np.max(res)))
		print("Min is {}".format(np.min(res)))
		with open("parameters.csv", 'a') as f:
			msg = "{},{},{},{},{},{},{},{}\n".format(self.Ne, self.C, self.gamma, avg, np.max(res), np.min(res), self.train_eps, self.test_eps)
			f.write(msg)

	def excute(self):
		self.train()
		self.test()

def main():
	# Ne, C, gamma
	parameters = (40, 40, 0.7)
	app = Application(parameters, train_eps=10000, test_eps=1000)
	app.excute()

if __name__ == "__main__":
	main()