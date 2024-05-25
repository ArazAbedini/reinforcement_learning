import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from model import Model
import gym


env = gym.make('CartPole-v1', render_mode='rgb_array')
obs, info = env.reset(seed=42)



FILE_PATH = 'data/AVAX_1h.txt'
model = Model(FILE_PATH)
# print(model.df)
# model.plot_df()
print(model.average_derivation(5))
model.smooth_filter(21,3)
# model.find_maximumm()
# model.plot_df()
# close_array = np.array([3, 5, 2, 8, 7, 6, 5, 9, 10, 3, 6, 2, 5, 8, 9, 1, 4, 7, 2, 6, 3,3, 5, 2, 8, 7, 6, 5, 9, 10, 3, 6, 2, 5, 8, 9, 1, 4, 7, 2, 6, 3])
# model.assign_states(close_array, 21, 3)
model.Q_learning_process()
print(model.Q_table)