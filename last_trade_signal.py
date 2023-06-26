import pandas as pd
import numpy as np
import scipy
import pandas_ta as ta
from donchian import donchian_breakout, get_trades_from_signal
import matplotlib.pyplot as plt


def last_trade_adj_signal(ohlc: pd.DataFrame, signal: np.array, last_winner: bool = False):
    # Input signal must be long and short, only having values of 1 and -1.
    # Adjust a signal to only trade if last trade was a winner/loser

    last_type = -1
    if last_winner:
        last_type = 1
    
    close = ohlc['close'].to_numpy()
    mod_signal = np.zeros(len(signal))

    long_entry_p = np.nan
    short_entry_p = np.nan
    last_long = np.nan
    last_short = np.nan

    last_sig = 0.0
    for i in range(len(close)):
        if signal[i] == 1.0 and last_sig != 1.0: # Long entry
            long_entry_p = close[i]
            if not np.isnan(short_entry_p):
                last_short = np.sign(short_entry_p - close[i])
                short_entry_p = np.nan

        if signal[i] == -1.0  and last_sig != -1.0: # Short entry
            short_entry_p = close[i]
            if not np.isnan(long_entry_p):
                last_long = np.sign(close[i] - long_entry_p)
                long_entry_p = np.nan
        
        last_sig = signal[i]
        
        if signal[i] == 1.0 and last_short == last_type:
            mod_signal[i] = 1.0
        if signal[i] == -1.0 and last_long == last_type:
            mod_signal[i] = -1.0
        
    return mod_signal



if __name__ == '__main__':


    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    data = data.dropna()

    
    data['r'] = np.log(data['close']).diff().shift(-1)
    donchian_breakout(data, 24)
    data['last_lose'] = last_trade_adj_signal(data, data['signal'].to_numpy(), last_winner=False)
    data['last_win'] = last_trade_adj_signal(data, data['signal'].to_numpy(), last_winner=True)
   
    orig = data['r'] * data['signal']
    lose = data['r'] * data['last_lose']
    win = data['r'] * data['last_win']
    print("Original PF", orig[orig > 0].sum() / orig[orig < 0].abs().sum())
    print("Last Lose PF", lose[lose > 0].sum() / lose[lose < 0].abs().sum())
    print("Last Win PF", win[win > 0].sum() / win[win < 0].abs().sum())

    plt.style.use('dark_background')
    orig.cumsum().plot(label='All Trades')
    lose.cumsum().plot(label='Last Loser')
    win.cumsum().plot(label='Last Winner')
    plt.legend()
    plt.show()


    # Compute across many lookbacks
    lookbacks = list(range(12, 169, 6))
    pfs = []
    types = []
    lbs = []
    for lookback in lookbacks:
        donchian_breakout(data, lookback)
        data['last_lose'] = last_trade_adj_signal(data, data['signal'].to_numpy(), last_winner=False)
        data['last_win'] = last_trade_adj_signal(data, data['signal'].to_numpy(), last_winner=True)
        
        orig = data['r'] * data['signal']
        lose = data['r'] * data['last_lose']
        win = data['r'] * data['last_win']

        pfs.append ( np.log( orig[orig > 0].sum() / orig[orig < 0].abs().sum() ) )
        lbs.append(lookback)
        types.append("All")
        
        pfs.append (np.log( lose[lose > 0].sum() / lose[lose < 0].abs().sum() ) )
        lbs.append(lookback)
        types.append("Last Loser")
        
        pfs.append ( np.log( win[win > 0].sum() / win[win < 0].abs().sum() ) )
        lbs.append(lookback)
        types.append("Last Winner")
    

    import seaborn as sns 
    df = pd.DataFrame()
    df['Lookback'] = lbs
    df['Type'] = types
    df['Log(Profit Factor)'] = pfs
    
    plt.style.use('dark_background')
    sns.catplot(
        data=df, y="Log(Profit Factor)", x='Lookback', hue="Type", kind='bar',
        palette="dark", edgecolor=".6", legend=False
    )
    plt.axhline(0.0, color='white')
    plt.legend(prop={'size': 16}, title='Signal Type')
    plt.show()







