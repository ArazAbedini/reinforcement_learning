import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast


class Model():
    def __init__(self, file_path):
        self.actions = [-1, 0, 1]
        self.states = ['x_buy', 'buy', 'hold', 'sell', 'x_sell']
        self.df = pd.DataFrame()
        self.read_file(file_path)




    def read_file(self, file_path: str) -> pd.DataFrame:
        FILE_PATH = file_path
        with open(FILE_PATH, 'r') as file:
            content = file.read()
        content_list = ast.literal_eval(content)
        new_df = pd.DataFrame(content_list)
        self.df = new_df


    def plot_df(self):
        fig, ax = plt.subplots()
        int_id = self.df['id'].astype('uint16')
        float_price = self.df['close'].astype('float16')
        id_array = np.array(int_id)
        price_array = np.array(float_price)
        ax.plot(id_array, price_array, label='overall price')
        plt.show()

        # print(x.info)



