# CS 440 MP4 Part-1: Q learning

## Implement:
In this part of the assignment, you will create a snake agent to learn how to get food as many as possible without dying. In order to do this, you must use Q-learning. Implement the TD Q-learning algorithm.

## Requirements:
```
python3
pygame
```
## Running:
The main file to run the mp is *snake_main.py*:

```
usage: snake_main.py [-h] [--human] [--model_name MODEL_NAME]
                     [--train_episodes TRAIN_EPS] [--test_episodes TEST_EPS]
                     [--show_episodes SHOW_EPS] [--window WINDOW] [--Ne NE]
                     [--C C] [--gamma GAMMA] [--snake_head_x SNAKE_HEAD_X]
                     [--snake_head_y SNAKE_HEAD_Y] [--food_x FOOD_X]
                     [--food_y FOOD_Y]

optional arguments:
  -h, --help            show this help message and exit
  --human               making the game human playable - default False
  --model_name MODEL_NAME
                        name of model to save if training or to load if
                        evaluating - default q_agent
  --train_episodes TRAIN_EPS
                        number of training episodes - default 10000
  --test_episodes TEST_EPS
                        number of testing episodes - default 1000
  --show_episodes SHOW_EPS
                        number of displayed episodes - default 10
  --window WINDOW       number of episodes to keep running stats for during
                        training - default 100
  --Ne NE               the Ne parameter used in exploration function -
                        default 40
  --C C                 the C parameter used in learning rate - default 40
  --gamma GAMMA         the gamma paramter used in learning rate - default 0.7
  --snake_head_x SNAKE_HEAD_X
                        initialized x position of snake head - default 200
  --snake_head_y SNAKE_HEAD_Y
                        initialized y position of snake head - default 200
  --food_x FOOD_X       initialized x position of food - default 80
  --food_y FOOD_Y       initialized y position of food - default 80
```
