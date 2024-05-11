import numpy as np


transition_probabilities = [ # shape=[s, a, s']
[[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
[[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
[None, [0.8, 0.1, 0.1], None]
]
rewards = [ # shape=[s, a, s']
[[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
[[0, 0, 0], [0, 0, 0], [0, 0, -50]],
[[0, 0, 0], [+40, 0, 0], [0, 0, 0]]
]
possible_actions = [[0, 1, 2], [0, 2], [1]]


Q_values = np.full((3, 3), -np.inf) # -np.inf for impossible actions
for state, actions in enumerate(possible_actions):
    Q_values[state, actions] = 0.0


gamma = 0.90 # the discount factor
for iteration in range(50):
    Q_prev = Q_values.copy()
    for s in range(3):
        for a in possible_actions[s]:
            t = 0
            for i in range(3):
                t = transition_probabilities[s][a][i] * (rewards[s][a][i] + gamma * Q_prev[i].max())
            Q_values[s, a] = t


def label_process(close: np.ndarray, open_time: np.ndarray) -> np.ndarray:
    fig, ax = plt.subplots()
    smooth_arr = signal.savgol_filter(close, window_length=14, polyorder=2)
    smooth_peak, _ = signal.find_peaks(x=smooth_arr)
    smooth_valleys, _ = signal.find_peaks(x=-1 * smooth_arr)
    smooth_peak = find_actual_peak(smooth_peak, close) # in this function we have index of peaks
    smooth_valleys = find_actual_valleys(smooth_valleys, close) # in this function we have index of valleys
    y_peaks = close[smooth_peak]
    y_valleys = close[smooth_valleys]
    label = np.zeros(len(close), dtype=np.int32)
    state = 0
    for i in range(len(close)):
        if i in smooth_peak:
            state = 1
            label[i] = 1
        elif i in smooth_valleys:
            state = 0
        elif state == 1:
            label[i] = 1

    # ax.plot(open_time, close, color='black')
    # ax.plot(open_time[smooth_peak], y_peaks, 'ro')
    # ax.plot(open_time[smooth_valleys], y_valleys, 'go')
    # plt.xticks(rotation=90)
    print(f'length of the smooth peak is {len(smooth_peak)}')
    print(f'length of close is {len(close)}')
    # plt.show()
    return label
