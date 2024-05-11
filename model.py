import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import numpy as np
import ast


class Model():
    def __init__(self, file_path):
        self.actions = [-1, 0, 1]
        self.states = ['x_buy', 'buy', 'hold', 'sell', 'x_sell']
        self.df = pd.DataFrame()
        self.read_file(file_path)
        self.price = self.calculate_price()
        self.smooth_price = self.smooth_filter(21, 3)
        self.peaks = self.find_maximumm()
        # self.find_maximumm()




    def read_file(self, file_path: str) -> pd.DataFrame:
        FILE_PATH = file_path
        with open(FILE_PATH, 'r') as file:
            content = file.read()
        content_list = ast.literal_eval(content)
        new_df = pd.DataFrame(content_list)
        self.df = new_df

    def calculate_price(self):
        return np.array(self.df['close'].astype('float16'))
        
    def plot_df(self):
        fig, ax = plt.subplots()
        int_id = self.df['id'].astype('uint16')
        price_array = self.price
        id_array = np.array(int_id)
        ax.plot(id_array, price_array, label='overall price')
        ax.scatter(self.peaks, self.price[self.peaks], color='red')
        ax.plot(id_array, self.smooth_price)
        plt.show()

    def find_maximumm(self):
        smooth_peaks, _ = signal.find_peaks(x=self.smooth_price)
        abs_peaks = []
        for point in smooth_peaks:
            start_index = max(0, point - 21)
            end_point = min(len(self.price) - 1, point + 21)
            max_value_index = np.argmax(self.price[start_index: end_point]) + start_index
            if max_value_index not in abs_peaks:
                abs_peaks.append(max_value_index)
        abs_array = np.array(abs_peaks).astype('int16')

        filtered_array = self.second_filter(abs_array)
        return filtered_array


    def second_filter(self, abs_array: np.array) -> np.array:
        bool_array = np.ones_like(abs_array, dtype=bool)
        for index in abs_array:
            condition = (index <= abs_array) & (abs_array < index + 21)
            price_index_list = abs_array[condition]
            indices = np.where(condition)[0]  # index of price_index_list in abs_array
            candidate_index = price_index_list[
                np.argmax(self.price[price_index_list])]  # its item in abs_array which is argmax
            candidate = np.where(price_index_list == candidate_index)[0][0]  # it is the index of candidate in indidces
            index_bool_array = np.zeros_like(indices, dtype=bool)
            index_bool_array[candidate] = True
            for index, bool_value in zip(indices, index_bool_array):
                if bool_value == False:
                    bool_array[index] = False
        second_filter = []
        for index, bool_value in zip(abs_array, bool_array):
            if bool_value == True:
                second_filter.append(index)
        
        return np.array(second_filter)
    
    
    
    def average_derivation(self, last_day: int) -> float:
        standard_start_day = last_day
        length_day = last_day
        price_array = self.price
        deriv_list = []
        for point in range(standard_start_day, len(self.df)):
            derivation = (price_array[point] - price_array[point -5]) / length_day
            deriv_list.append(derivation)

        derivation_array = np.abs(np.array(deriv_list))
        mean_value = derivation_array.mean()
        return mean_value
        
    def smooth_filter(self, length: int, polyorder: int):
        return np.array(signal.savgol_filter(self.price, length, polyorder))



