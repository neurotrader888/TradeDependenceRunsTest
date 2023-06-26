import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from donchian import donchian_breakout, get_trades_from_signal


def runs_test(signs: np.array):
    # signs must consist of only 1 and -1
    # Returns Z-Score of observed runs 
    assert len(signs) >= 2
    
    n_pos = len(signs[signs > 0])
    n_neg = len(signs[signs < 0])
    n = len(signs)
    
    # Mean number of expected runs
    mean = 2 * n_pos * n_neg / n  + 1
    # Stadard of expected runs
    std = (mean - 1) * (mean - 2) / ( n - 1 ) # Variance
    std = std ** 0.5

    #print(mean)
    #print(std**2)
    #print(std)
   
    # Count observed runs
    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1]:
            runs += 1 # Streak broken
    #print(runs)
    # Z-Score
    z = (runs - mean) / std
    return z


if __name__ == '__main__':


    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    data = data.dropna()
    
    donchian_breakout(data, 24)
    _,_,all_trades = get_trades_from_signal(data, data['signal'])
    signs = np.sign(all_trades['return']).to_numpy()
    runs_z = runs_test(signs)
    # For 24 we get 2.7...
    print("Donchian Breakout 24 Z-Score:", runs_z)

    all_runs_z = []
    lookbacks = list(range(12, 169, 2))
    for lookback in lookbacks:
        donchian_breakout(data, lookback)
        _,_,all_trades = get_trades_from_signal(data, data['signal'])
        signs = np.sign(all_trades['return']).to_numpy()
        runs_z = runs_test(signs)

        all_runs_z.append(runs_z)
    
    plt.style.use('dark_background')
    pd.Series(all_runs_z, index=lookbacks).plot()
    plt.xlabel("Lookback")
    plt.ylabel("Runs Test Z-Score")
    plt.show()






    


