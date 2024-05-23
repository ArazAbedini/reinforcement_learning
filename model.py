from scipy.stats import linregress
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import numpy as np
import random
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
        self.valleys = self.find_minimumm()
        self.Q_table = pd.DataFrame(data=0.0, index=self.states, columns=self.actions)







    def read_file(self, file_path: str) -> pd.DataFrame:
        FILE_PATH = file_path
        with open(FILE_PATH, 'r') as file:
            content = file.read()
        content_list = ast.literal_eval(content)
        new_df = pd.DataFrame(content_list)
        self.df = new_df
        print(len(self.df))

    def calculate_price(self):
        return np.array(self.df['close'].astype('float16'))
        
    def plot_df(self):
        fig, ax = plt.subplots()
        int_id = self.df['id'].astype('uint16')
        price_array = self.price
        id_array = np.array(int_id)
        ax.plot(id_array, price_array, label='overall price')
        ax.scatter(self.peaks, self.price[self.peaks], color='red')
        ax.scatter(self.valleys, self.price[self.valleys], color='green')
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

        filtered_array = self.second_filter(abs_array, 'max')
        return filtered_array

    def find_minimumm(self):
        smooth_valleys, _ = signal.find_peaks(x=-1 * self.smooth_price)
        abs_valleys = []
        for point in smooth_valleys:
            start_index = max(0, point - 21)
            end_point = min(len(self.price) - 1, point + 21)
            min_value_index = np.argmin(self.price[start_index: end_point]) + start_index
            if min_value_index not in abs_valleys:
                abs_valleys.append(min_value_index)
        abs_array = np.array(abs_valleys).astype('int16')

        filtered_array = self.second_filter(abs_array, 'min')
        return filtered_array


    def second_filter(self, abs_array: np.array, filter_type: str) -> np.array:
        bool_array = np.ones_like(abs_array, dtype=bool)
        for index in abs_array:
            condition = (index <= abs_array) & (abs_array < index + 21)
            price_index_list = abs_array[condition]
            indices = np.where(condition)[0]  # index of price_index_list in abs_array
            if filter_type == 'max':
                candidate_index = price_index_list[
                    np.argmax(self.price[price_index_list])]  # its item in abs_array which is argmax
            else:
                candidate_index = price_index_list[
                    np.argmin(self.price[price_index_list])]  # its item in abs_array which is argmax
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


    def assign_states(self, close_array: np.array,length: int, polyorder: int):

        diffs = np.diff(close_array)
        if np.all(diffs >= 0):
            if all(diffs[i] <= diffs[i + 1] for i in range(len(diffs) - 1)):
                return "x_sell"
            else:
                return "sell"
        elif np.all(diffs <= 0):
            if all(diffs[i] >= diffs[i + 1] for i in range(len(diffs) - 1)):
                return "x_buy"
            else:
                return "buy"
        else:
            smooth_array = signal.savgol_filter(close_array, length, polyorder)
            diffs = np.diff(smooth_array)
            if np.all(diffs >= 0):
                if all(diffs[i] <= diffs[i + 1] for i in range(len(diffs) - 1)):
                    return "x_sell"
                else:
                    return "sell"
            elif np.all(diffs <= 0):
                if all(diffs[i] >= diffs[i + 1] for i in range(len(diffs) - 1)):
                    return "x_buy"
                else:
                    return "buy"
            else:
                indices = np.arange(len(close_array))
                peaks, troughs = self.zigzag(close_array, 0.1)
                peak_indices, peak_values = zip(*peaks) if peaks else ([], [])
                trough_indices, trough_values = zip(*troughs) if troughs else ([], [])
                slope_peak, intercept_peak, r_peak, p_peak, std_err_peak = linregress(peak_indices, peak_values)
                slope_trough, intercept_trough, r_trough, p_trough, std_err_trough = linregress(trough_indices, trough_values)
                slope = (slope_peak + slope_trough) / 2
                intercept = (intercept_trough + intercept_peak) / 2
                if slope >= 2:
                    return "buy"
                elif slope <= -2:
                    return "sell"
                else:
                    return "hold"

    def zigzag(self, sequence: np.array, threshold=0.1):
        peaks = []
        troughs = []

        last_peak = sequence[0]
        last_trough = sequence[0]
        last_direction = 0  # 1 for peak, -1 for trough

        for i in range(1, len(sequence)):
            if last_direction == 0:  # initial case
                if sequence[i] > last_peak:
                    last_peak = sequence[i]
                    last_direction = 1
                    peaks.append((i, sequence[i]))
                elif sequence[i] < last_trough:
                    last_trough = sequence[i]
                    last_direction = -1
                    troughs.append((i, sequence[i]))
            elif last_direction == 1:
                if sequence[i] > last_peak:
                    last_peak = sequence[i]
                    peaks[-1] = (i, sequence[i])  # update the last peak
                elif sequence[i] <= last_peak * (1 - threshold):
                    last_trough = sequence[i]
                    last_direction = -1
                    troughs.append((i, sequence[i]))
            elif last_direction == -1:
                if sequence[i] < last_trough:
                    last_trough = sequence[i]
                    troughs[-1] = (i, sequence[i])  # update the last trough
                elif sequence[i] >= last_trough * (1 + threshold):
                    last_peak = sequence[i]
                    last_direction = 1
                    peaks.append((i, sequence[i]))

        return peaks, troughs


    def Q_learning_process(self):
        alpha = 0.1
        gamma = 0.9
        epsilon = 0.1
        visited = set()
        for i in range(1000):
            end_point = random.randint(21, 28251)
            while end_point in visited:
                end_point = random.randint(21, 28251)
            close_array = np.array(self.price[end_point - 20:end_point + 1])
            state = self.assign_states(close_array, 21, 3)
            if random.random() < epsilon:
                action = random.choice(self.actions)
            else:
                action = self.Q_table.loc[state].idxmax()




