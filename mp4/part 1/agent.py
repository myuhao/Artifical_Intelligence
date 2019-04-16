import numpy as np
import utils
import random


class Agent:

    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

        self.reset()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        Rewards: +1 when your action results in getting the food (snake head position is the same as the food position), -1 when the snake dies, that is when snake head hits the wall, its body segment or the head tries to move towards its adjacent body segment (moving backwards). -0.1 otherwise (does not die nor get food).
        '''

        # Index of the current state.
        # s' in the equation
        curr_state_idx = self._discretizeState(state)

        # If testing, just return the action with max Q value with respect to the current state, use exploitation and no exploration.
        if not self._train:
            return self.actions[self._myargmax(self.Q[curr_state_idx])]

        # Handle the first state, does not update anything, just return the best action based on the Q_table.
        if self.s == None:
            self.s = state
            self.a = self.actions[self._myargmax(self._explorationFunc(curr_state_idx))]
            self.points = points
            return self.a

        # Index of the pervious state stored in self.s
        # s in the equation
        prev_state_idx = self._discretizeState(self.s)

        # Reward
        R_s = 0
        if dead:
            R_s = -1
        elif points > self.points:
            R_s = +1
        else:
            R_s = -0.1

        # Update the Q_table.
        alpha = self.C /(self.C + self.N[prev_state_idx][self.a])
        max_expected = np.max(self.Q[curr_state_idx])
        self.Q[prev_state_idx][self.a] = self.Q[prev_state_idx][self.a] + alpha * (R_s + self.gamma * max_expected - self.Q[prev_state_idx][self.a])

        # if dead:
        #     self.reset()
        #     return self.a

        # self.N[prev_state_idx][self.a] += 1

        # Get the next action.
        action = self.actions[self._myargmax(self._explorationFunc(curr_state_idx))]
        self.a = action
        self.s = state
        self.points = points

        self.N[curr_state_idx][action] += 1

        return action

    def _discretizeState(self, state):
        '''
        Discretize the state tuple into the state configuration in the Q_table.
        Arguments:
            state -- a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        Returns:
            tuple -- Index of the actions scores in the Q_table.
        '''
        snake_head_x, snake_head_y, snake_body, food_x, food_y = state

        # Indeice of the state to be looked up in the Q_table.
        idx_adj_wall_x      = 0
        idx_adj_wall_y      = 0
        idx_food_dir_x      = 0
        idx_food_dir_y      = 0
        idx_adj_body_top    = 0
        idx_adj_body_bottom = 0
        idx_adj_body_left   = 0
        idx_adj_body_right  = 0

        # Check if there is wall next to the snake head.
        # Used logic in snake.py line 169-170.
        if snake_head_x == utils.GRID_SIZE:
            idx_adj_wall_x = 1
        elif snake_head_x == utils.DISPLAY_SIZE - utils.GRID_SIZE - utils.GRID_SIZE:
            idx_adj_wall_x = 2
        else:
            idx_adj_wall_x = 0
        if snake_head_y == utils.GRID_SIZE:
            idx_adj_wall_y = 1
        elif snake_head_y == utils.DISPLAY_SIZE - utils.GRID_SIZE - utils.GRID_SIZE:
            idx_adj_wall_y = 2
        else:
            idx_adj_wall_y = 0

        # food_coord - head_coord
        delta_food_x = food_x - snake_head_x
        delta_food_y = food_y - snake_head_y
        if delta_food_x > 0:
            # food_x right to head_x
            idx_food_dir_x = 2
        elif delta_food_x < 0:
            # food_x left to head_x
            idx_food_dir_x = 1
        else:
            # food_x == head_x
            idx_food_dir_x = 0
        if delta_food_y > 0:
            # food_y bottom to head_y
            idx_food_dir_y = 2
        elif delta_food_y < 0:
            # food_y top to head_y
            idx_food_dir_y = 1
        else:
            # food_y == head_y
            idx_food_dir_y = 0

        # Coordinates of the neighboring cells to the snake head.
        # Use utils.GRID_SIZE.
        top    = (snake_head_x, snake_head_y - utils.GRID_SIZE)
        bottom = (snake_head_x, snake_head_y + utils.GRID_SIZE)
        left   = (snake_head_x - utils.GRID_SIZE, snake_head_y)
        right  = (snake_head_x + utils.GRID_SIZE, snake_head_y)

        # Check adjoining snake body.
        for body_coord in snake_body:
            if body_coord == top:
                idx_adj_body_top = 1
            if body_coord == bottom:
                idx_adj_body_bottom = 1
            if body_coord == left:
                idx_adj_body_left = 1
            if body_coord == right:
                idx_adj_body_right = 1

        # Return the indices as a tuple.
        return (idx_adj_wall_x, idx_adj_wall_y, idx_food_dir_x, idx_food_dir_y, idx_adj_body_top, idx_adj_body_bottom, idx_adj_body_left, idx_adj_body_right)

    def _explorationFunc(self, state_idx):
        '''
        Based on the current state, get the scores of all the four actions using the exploration
        function listed.
        Arguments:
            state       -- a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        Returns:
            np.ndarray  -- Scores of each action.
        '''
        R_plus = 1
        u = self.Q[state_idx].copy()
        n = self.N[state_idx]
        u[n < self.Ne] = R_plus
        return u

    def _myargmax(self, a, axis=None, out=None):
        '''
        Get the max value index in the array.
        If value is the same, use the one with largest index value.
        '''
        idx = 0
        currMax = -np.inf
        for i in range(len(a)):
            if a[i] >= currMax:
                idx = i
                currMax = a[i]
        return idx


if __name__ == "__main__":
    tuple1 = (1, 1)
    print(tuple1 == (1,0))
