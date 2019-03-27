import random
import numpy as np
import tensorflow as tf

result = []
for i in range(10):
    result.append(i)

random.shuffle(result)

print(result)

a = np.zeros((784+1, 10))
print(a)
