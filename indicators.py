''' I N D I C A T O R S '''

'''
Some Trading Indicators Build From Scratch
'''

'''
1 ] Keltner Channels
2 ] ATR - Avarage True Length
3 ] Stochastic
4 ] Bollinger Bands
5 ] RSI - Relative Strength Index
6 ] MACD - Moving Avarage Convergence Divergence
7 ] Liquidity Pool Area
8 ] Upper Nadaraya-Watson envelope
9 ] Lower Nadaraya-Watson envelope
10 ] Channel and its position
11 ] Supply & Demand by Volume
12 ] Support & Resistance + Volume
13 ] Supply & Demand by Candles
14 ] Support & Resistance
15 ] VWAP - Volume-Weighted Average Daily reset
16 ] Mountain Nort, Mountain South, Higher-High & Lower-Low
17 ] Slope
18 ] Trend


'''


import pandas as pd
import numpy as np



# Keltner Channels
# --------------------------------------------------------------------------- #
def KeltnerChannels(df: pd.DataFrame, n_ema=20, n_atr=10):
    df['EMA'] = df.close.ewm(span=n_ema, min_periods=n_ema).mean()
    df = ATR(df, Length=n_atr)
    df['KCH_Up'] = df.ATR * 2 + df.EMA
    df['KCH_Lo'] = df.EMA - df.ATR * 2
    df.drop('ATR', axis=1, inplace=True)

    return df


# ATR - Avarage True Range
# --------------------------------------------------------------------------- #
def ATR(df: pd.DataFrame, Length=8):
    prev_close = df.close.shift(1)
    true_range_1 = df.high - df.low
    true_range_2 = abs(df.high - prev_close)
    true_range_3 = abs(prev_close - df.low)
    tr = pd.DataFrame({'Tr_1':true_range_1, 'Tr_2':true_range_2, 'Tr_3':true_range_3}).max(axis=1)
    df['ATR'] = tr.rolling(window=Length).mean()

    return df


# Stochastic
# --------------------------------------------------------------------------- #
def Stochastich(df: pd.DataFrame):
    df['20-high'] = df['high'].rolling(20).max()
    df['20-low'] = df['low'].rolling(20).min()
    df['%K'] = (df['close'].rolling(2).mean() - df['20-low'])*100/(df['20-high'] - df['20-low'])
    df['%D'] = df['%K'].rolling(3).mean()
    # %D == SLOW  %k == FAST
    df['FAST'] = df['%K']
    df['SLOW'] = df['%D']

    return df


# Bollinger Bands
# --------------------------------------------------------------------------- #
def BollingerBands(df: pd.DataFrame, n=20, s=2):
    typical_p = (df.close + df.high + df.low) / 3
    stddev = typical_p.rolling(window=n).std()
    df['BB_MA'] = typical_p.rolling(window=n).mean()
    df['BB_UP'] = df['BB_MA'] + stddev * s
    df['BB_LO'] = df['BB_MA'] - stddev * s

    return df


# RSI - Relative Strength Index
# --------------------------------------------------------------------------- #
def RSI(df: pd.DataFrame, n=14):
    alpha = 1.0 / n
    gains = df.close.diff()

    wins = pd.Series([x if x >= 0 else 0.0 for x in gains], name='wins')
    losses = pd.Series([x * -1 if x < 0 else 0.0 for x in gains], name='losses')

    wins_rma = wins.ewm(min_periods=n, alpha=alpha).mean()
    losses_rma = losses.ewm(min_periods=n, alpha=alpha).mean()
    rs = wins_rma / losses_rma

    rs = wins_rma / losses_rma

    df[f'RSI_{n}'] = 100.0 - (100.0 / (1.0 + rs))
    # df.reset_index(drop=True, inplace=True)
    return df


# MACD - Moving Avarage Convergence Divergence
# --------------------------------------------------------------------------- #
def MACD(df: pd.DataFrame, n_slow=26, n_fast=12, n_signal=9):
    ema_long = df.close.ewm(min_periods=n_slow, span=n_slow).mean()
    ema_short = df.close.ewm(min_periods=n_fast, span=n_fast).mean()

    df['MACD'] = ema_short - ema_long
    df['SIGNAL'] = df.MACD.ewm(min_periods=n_signal, span=n_signal).mean()
    df['HIST'] = df.MACD - df.SIGNAL

    return df


# LIQUIDITY POOL
# --------------------------------------------------------------------------- #
def Liquidity_pool(df: pd.DataFrame, Length=22):
    # LIQUIDITY
    df['Liquidity'] = (df['close'] * df['volume']) / df['volume'].rolling(window=Length).sum()
    df['LIQ'] = df.Liquidity * 100

    return df


# Upper Nadaraya-Watson envelope 
# --------------------------------------------------------------------------- #
def upper_envelope(df: pd.DataFrame, Size = 2.2):
    H_bandwidth = Size
    H_x = np.array(df.index)
    H_y = np.array(df['high'])
    H_weights = np.exp(-0.5 * (H_x[:, None] - H_x[None, :])**2 / H_bandwidth**2)
    H_weights /= np.sum(H_weights, axis=1)[:, None]
    H_smoothed = np.sum(H_weights * H_y[None, :], axis=1)
    df['upper_envelope'] = H_smoothed + 2 * np.std(H_weights * (H_y[None, :] - H_smoothed[:, None]), axis=1)

    return df


# Lower Nadaraya-Watson envelope
# --------------------------------------------------------------------------- #
def lower_envelope(df: pd.DataFrame, Size = 2.2):
    bandwidth = Size
    x = np.array(df.index)
    y = np.array(df['low'])
    weights = np.exp(-0.5 * (x[:, None] - x[None, :])**2 / bandwidth**2)
    weights /= np.sum(weights, axis=1)[:, None]
    smoothed = np.sum(weights * y[None, :], axis=1)
    df['lower_envelope'] = smoothed - 2 * np.std(weights * (y[None, :] - smoothed[:, None]), axis=1)

    return df


# Channel and its position
# --------------------------------------------------------------------------- #
def position_in_channel(df: pd.DataFrame, Length=8, Shift=1):
    df['UP'] = df.close.rolling(window=Length).max().shift(Shift)
    df['LO'] = df.close.rolling(window=Length).min().shift(Shift)
    df['position_in_channel'] = (df['close'] - df['LO']) / (df['UP'] - df['LO'])

    return df


# Supply & Demand by Volume
# --------------------------------------------------------------------------- #
def Supply_Demand_Volume(df: pd.DataFrame, Roll_Vwap = 44, Roll_Range = 70, Range_Mean = 3):
    df["range"] = (df["high"] - df["low"])
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["vwap_H"] = (df["high"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["vwap_L"] = (df["low"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df['RollRange'] = df["range"].rolling(window=Roll_Range).mean()
    df["RollVwap"] = df["vwap"].rolling(window=Roll_Vwap).mean()
    df['range_mean'] = df.range.rolling(window=Range_Mean).mean()

    return df


# Supply & Demand by Candles
# --------------------------------------------------------------------------- #
def Supply_Deamnd_Candles(df: pd.DataFrame, Length=40):
    df["high_rolling_max_20"] = df["high"].rolling(window=Length).max()#.shift(1)
    df["low_rolling_min_20"] = df["low"].rolling(window=Length).min()#.shift(1)
    # Create a column to store the supply and demand zones
    df["supply_demand_std"] = None

    for i, row in df.iterrows():
        if row["high"] >= row["high_rolling_max_20"]:
            df.loc[i, "supply_demand_std"] = "Supply"
        elif row["low"] <= row["low_rolling_min_20"]:
            df.loc[i, "supply_demand_std"] = "Demand"
        else:
            df.loc[i, "supply_demand_std"] = 'Nope'

    return df


# Support & Resistance + Volume
# --------------------------------------------------------------------------- #
def Support_Resistance_Volume(df: pd.DataFrame, Highs_Range=40, Lows_Range=40, Volume_Range=40, Shift=2):
    highs = df['high'].rolling(window=Highs_Range).max().shift(Shift)
    lows = df['low'].rolling(window=Lows_Range).min().shift(Shift)
    # Calculate rolling average volume
    volume = df['volume'].rolling(window=Volume_Range).mean().shift(Shift)
    # Calculate support and resistance levels based on price and volume
    df['support_volume'] = lows + ((highs - lows) * volume / volume.max())
    df['resistance_volume'] = highs - ((highs - lows) * volume / volume.max())

    return df


# Support & Resistance
# --------------------------------------------------------------------------- #
def Support_Resistance(df: pd.DataFrame, Length=40, Shift=1):
    df["resistance"] = df["close"].rolling(window=Length).max().shift(Shift)
    df["support"] = df["close"].rolling(window=Length).min().shift(Shift)

    return df


# Volume-Weighted Average (VWAP) Daily reset
# --------------------------------------------------------------------------- #
def v_wap(df: pd.DataFrame):
    df['vwap'] = np.cumsum(df['volume'] * (df['high'] + df['low'] + df['close']) / 3) / np.cumsum(df['volume'])
    df['dev'] = np.sqrt(np.cumsum(df['volume'] *    ((df['close'] - df['vwap']) ** 2 + 
                                                    (df['high'] - df['vwap']) ** 2 + 
                                                    (df['low'] - df['vwap']) ** 2)) / np.cumsum(df['volume']))
    df['time'] = pd.to_datetime(df['time'])
    df = df.groupby(df.time.dt.date).apply(v_wap)

    return df


# Mountain Nort, Mountain South, Higher-High & Lower-Low
# --------------------------------------------------------------------------- #
def Mountain_North_South(df: pd.DataFrame):
    df['South'] = np.nan
    df['North'] = np.nan
    for i in range(5, len(df) - 5):
        if      (df['low'][i] <= df['low'][i-1]) \
            and (df['low'][i-1] <= df['low'][i-2]) \
            and (df['low'][i-2] <= df['low'][i-3]) \
            and (df['low'][i-3] <= df['low'][i-4]) \
            and (df['low'][i-4] <= df['low'][i-5]) \
            and (df['low'][i] <= df['low'][i+1]) \
            and (df['low'][i+1] <= df['low'][i+2]) \
            and (df['low'][i+2] <= df['low'][i+3]) \
            and (df['low'][i+3] <= df['low'][i+4]) \
            and (df['low'][i+4] <= df['low'][i+5]):

            df.loc[i, 'South'] = df.loc[i, 'low']

        if      (df['high'][i] >= df['high'][i-1]) \
            and (df['high'][i-1] >= df['high'][i-2]) \
            and (df['high'][i-2] >= df['high'][i-3]) \
            and (df['high'][i-3] >= df['high'][i-4]) \
            and (df['high'][i-4] >= df['high'][i-5]) \
            and (df['high'][i] >= df['high'][i+1]) \
            and (df['high'][i+1] >= df['high'][i+2]) \
            and (df['high'][i+2] >= df['high'][i+3]) \
            and (df['high'][i+3] >= df['high'][i+4]) \
            and (df['high'][i+4] >= df['high'][i+5]):

            df.loc[i, 'North'] = df.loc[i, 'high']
    df['South'].fillna(0, inplace=True)
    df['North'].fillna(0, inplace=True)

    return df


# Slope
# --------------------------------------------------------------------------- #
def Slope(df: pd.DataFrame, N=8):
    df["range"] = (df["high"] - df["low"])
    prices = df['close'].values
    n = N
    array_sl = [j*0 for j in range(n-1)]
    for j in range(n, len(prices) + 1):
        y = prices[j-n:j]
        x = np.array(range(n))
        x_sc = (x - x.min()) / (x.max() - x.min())
        y_sc = (y - y.min()) / (y.max() - y.min())
        x_sc = np.column_stack((np.ones(n), x_sc))
        b = np.linalg.inv(x_sc.T.dot(x_sc)).dot(x_sc.T).dot(y_sc)
        array_sl.append(b[-1])
    slope_angle = np.rad2deg(np.arctan(np.array(array_sl)))
    df['slope'] = (slope_angle * np.pi / 180 * df['range'].mean() / n) * 10 # or * (38+(80/100)

    return df


# Trend
# --------------------------------------------------------------------------- #
def Trend(df: pd.DataFrame, Length=50, n_candles=6):
    df['Simple_Moving_Avarage'] = df.close.rolling(window=Length).mean()
    # df['Ex_Moving_Avarage'] = df.ewm(span=Length, adjust=True).mean() # uncomment this one for EMA
    Mov_Ava = [0] * len(df)
    backcandles = n_candles
    for row in range(backcandles, len(df)):
        upt = 1 # uptrend
        dnt = 1 # downtrend
        for i in range(row-backcandles, row+1):
            if max(df['open'][i], df['close'][i]) >= df['Simple_Moving_Avarage'][i]:
                dnt = 0
            if max(df['open'][i], df['close'][i]) <= df['Simple_Moving_Avarage'][i]:
                upt = 0
        if upt == 1 and dnt == 1:
            Mov_Ava[row] = 3
        if upt == 1:
            Mov_Ava[row] = 2 # up
        if dnt == 1:
            Mov_Ava[row] = 1 # down
    df['Trend_SMA'] = Mov_Ava

    return df








