from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
def zigzag(sequence, threshold):
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


# Sample sequence of 42 numbers
sequence = np.array([3, 5, 2, 8, 7, 6, 5, 9, 10, 3, 6, 2, 5, 8, 9, 1, 4, 7, 2, 6, 3,
                     2, 3, 4, 6, 7, 5, 3, 2, 4, 6, 8, 5, 7, 9, 4, 5, 6, 2, 4, 1])

# Define the threshold for the Zig Zag indicator
threshold = 0.1  # 10%

# Calculate peaks and troughs
peaks, troughs = zigzag(sequence, threshold)

# Plot the sequence
indices = np.arange(len(sequence))
plt.plot(indices, sequence, marker='o', label='Sequence')

# Plot the Zig Zag indicator
peak_indices, peak_values = zip(*peaks) if peaks else ([], [])
trough_indices, trough_values = zip(*troughs) if troughs else ([], [])
slope_peak, intercept_peak, r, p, std_err = stats.linregress(peak_indices, peak_values)
slope_trough, intercept_trough, r, p, std_err = stats.linregress(trough_indices, trough_values)

def line_peak(x):
    return slope_peak * x + intercept_peak

def line_through(x):
    return slope_trough * x + intercept_trough

def line_mean(x):
    return ((slope_peak + slope_trough) / 2) * x + (intercept_peak + intercept_trough) / 2

plt.plot(indices, line_through(indices))
plt.plot(indices, slope_peak * indices + intercept_peak)
plt.plot(indices, line_mean(indices))
plt.plot(peak_indices, peak_values, color='green', marker='o', linestyle='-', label='Zig Zag Peaks')
plt.plot(trough_indices, trough_values, color='red', marker='o', linestyle='-', label='Zig Zag Troughs')

plt.show()