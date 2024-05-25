from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'A': [1, 5, 3],
    'B': [4, 2, 6],
    'C': [7, 8, 5]
}, index=['first', 'second', 'third'])


print(df)

print(np.max(df.loc['second']))