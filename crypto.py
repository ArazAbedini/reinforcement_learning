import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import gym


env = gym.make('CartPole-v1', render_mode='rgb_array')
obs, info = env.reset(seed=42)

FILE_PATH = '/home/araz/Documents/ai/files/AVAX_1h.txt'
with open(FILE_PATH, 'r') as file:
    content = file.read()
content_list = ast.literal_eval(content)