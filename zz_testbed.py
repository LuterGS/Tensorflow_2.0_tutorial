import gym
import random
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
env.reset()

print(np.eye(16)[3])


test = [
    [1, 2, 3],
    [4, 5, 6]
]

test = list(np.array(test))

for i in range(2):
    test[i] = sum(test[i])

print(test)