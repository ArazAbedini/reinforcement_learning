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
model.plot_df()

