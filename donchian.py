import pandas as pd
import numpy as np
import scipy
import pandas_ta as ta
import matplotlib.pyplot as plt

def donchian_breakout(df: pd.DataFrame, lookback: int):
    # input df is assumed to have a 'close' column
    df['upper'] = df['close'].rolling(lookback - 1).max().shift(1)
    df['lower'] = df['close'].rolling(lookback - 1).min().shift(1)
    df['signal'] = np.nan
    df.loc[df['close'] > df['upper'], 'signal'] = 1
    df.loc[df['close'] < df['lower'], 'signal'] = -1
    df['signal'] = df['signal'].ffill()

def get_trades_from_signal(data: pd.DataFrame, signal: np.array):
    # Gets trade entry and exit times from a signal
    # that has values of -1, 0, 1. Denoting short,flat,and long.
    # No position sizing.

    long_trades = []
    short_trades = []

    close_arr = data['close'].to_numpy()
    last_sig = 0.0
    open_trade = None
    idx = data.index
    for i in range(len(data)):
        if signal[i] == 1.0 and last_sig != 1.0: # Long entry
            if open_trade is not None:
                open_trade[2] = idx[i]
                open_trade[3] = close_arr[i]
                short_trades.append(open_trade)

            open_trade = [idx[i], close_arr[i], -1, np.nan]
        if signal[i] == -1.0  and last_sig != -1.0: # Short entry
            if open_trade is not None:
                open_trade[2] = idx[i]
                open_trade[3] = close_arr[i]
                long_trades.append(open_trade)

            open_trade = [idx[i], close_arr[i], -1, np.nan]
        
        if signal[i] == 0.0 and last_sig == -1.0: # Short exit
            open_trade[2] = idx[i]
            open_trade[3] = close_arr[i]
            short_trades.append(open_trade)
            open_trade = None

        if signal[i] == 0.0  and last_sig == 1.0: # Long exit
            open_trade[2] = idx[i]
            open_trade[3] = close_arr[i]
            long_trades.append(open_trade)
            open_trade = None

        last_sig = signal[i]

    long_trades = pd.DataFrame(long_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])
    short_trades = pd.DataFrame(short_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])

    long_trades['return'] = (long_trades['exit_price'] - long_trades['entry_price']) / long_trades['entry_price']
    short_trades['return'] = -1 * (short_trades['exit_price'] - short_trades['entry_price']) / short_trades['entry_price']
    long_trades = long_trades.set_index('entry_time')
    short_trades = short_trades.set_index('entry_time')
    
    long_trades['type'] = 1
    short_trades['type'] = -1
    all_trades = pd.concat([long_trades, short_trades])
    all_trades = all_trades.sort_index()
    
    return long_trades, short_trades, all_trades



if __name__ == '__main__':


    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    data = data.dropna()

    
    data['r'] = np.log(data['close']).diff().shift(-1)
    donchian_breakout(data, 24)
    plt.style.use('dark_background')
    
    '''
    data['close'].plot()
    data['upper'].plot(color='green')
    data['lower'].plot(color='red')
    plt.twinx()
    data['signal'].plot(color='orange')
    plt.show()
    '''

    long_trades, short_trades, all_trades = get_trades_from_signal(data, data['signal']) 
    all_trades['return'].hist(bins=50)
    plt.xlabel("Trade Return %")
    plt.ylabel("# Of Trades")
    plt.show()

    rets = data['r'] * data['signal']
    pf = rets[rets>0].sum() / rets[rets<0].abs().sum()
    print("Profit Factor", pf)
    print("Avg Trade", all_trades['return'].mean())
    print("Win Rate", len(all_trades[all_trades['return'] > 0]) / len(all_trades))

    (data['r'] * data['signal']).cumsum().plot()
    plt.show()
   


    all_trades['lag1'] = all_trades['return'].shift(1)
    all_trades.plot.scatter('lag1', 'return')
    





