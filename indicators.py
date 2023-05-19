''' I N D I C A T O R S '''

'''
Keltner Channels
ATR - Avarage True Range
Bollinger Bands
RSI - Relative Strength Index
MACD - Moving Avarage Convergence Divergence

 '''


import pandas as pd





# Keltner Channels
def KeltnerChannels(df: pd.DataFrame, n_ema=20, n_atr=10):
    df['EMA'] = df.mid_c.ewm(span=n_ema, min_periods=n_ema).mean()
    df = ATR(df, n=n_atr)
    df['KCH_Up'] = df.ATR * 2 + df.EMA
    df['KCH_Lo'] = df.EMA - df.ATR * 2
    df.drop('ATR', axis=1, inplace=True)
    return df


# ATR - Avarage True Range
def ATR(df: pd.DataFrame, n=14):
    prev_close = df.mid_c.shift(1)
    true_range_1 = df.mid_h - df.mid_l
    true_range_2 = abs(df.mid_h - prev_close)
    true_range_3 = abs(prev_close - df.mid_l)

    tr = pd.DataFrame({'Tr_1':true_range_1, 'Tr_2':true_range_2, 'Tr_3':true_range_3}).max(axis=1)
    df['ATR'] = tr.rolling(window=n).mean()
    return df


# Bollinger Bands
def BollingerBands(df: pd.DataFrame, n=20, s=2):
    typical_p = (df.mid_c + df.mid_h + df.mid_l) / 3
    stddev = typical_p.rolling(window=n).std()
    df['BB_MA'] = typical_p.rolling(window=n).mean()
    df['BB_UP'] = df['BB_MA'] + stddev * s
    df['BB_LO'] = df['BB_MA'] - stddev * s
    return df


# RSI - Relative Strength Index
def RSI(df: pd.DataFrame, n=14):
    alpha = 1.0 / n
    gains = df.mid_c.diff()

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
def MACD(df: pd.DataFrame, n_slow=26, n_fast=12, n_signal=9):
    ema_long = df.mid_c.ewm(min_periods=n_slow, span=n_slow).mean()
    ema_short = df.mid_c.ewm(min_periods=n_fast, span=n_fast).mean()

    df['MACD'] = ema_short - ema_long
    df['SIGNAL'] = df.MACD.ewm(min_periods=n_signal, span=n_signal).mean()
    df['HIST'] = df.MACD - df.SIGNAL

    return df






