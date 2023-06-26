import pandas as pd
import numpy as np
import scipy
import pandas_ta as ta
from runs_test import runs_test
import matplotlib.pyplot as plt

# Didnt include in video, but I've seen talk online about
# using the runs test as an indicator
# Here I compute the runs test z-score using the signs of returns
# in a rolling window. I have not researched this much, idk if its any good.
# But its a well normalized indicator for yoru collection, have fun.  

def runs_trend_indicator(close: pd.Series, lookback: int):
    change_sign = np.sign(close.diff()).to_numpy()
    ind = np.zeros(len(close))
    ind[:] = np.nan

    for i in range(lookback, len(close)):
        ind[i] = runs_test(change_sign[i - lookback + 1: i+1])

    return ind

if __name__ == '__main__':
    

    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    data = data.dropna()

    data['runs_ind'] = runs_trend_indicator(data['close'], 24)


    



