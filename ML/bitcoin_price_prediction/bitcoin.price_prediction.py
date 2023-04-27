# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:50:37 2023

@author: Strange
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

# crypto_currency = 'BTC'
# against_currency = 'USD'

# start = dt.datetime(2016,1,1)
# end = dt.datetime.now()

# data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', start, end)

# '''preapre data '''
# print(data.head())

import ccxt
import pandas as pd

# Set the exchange and symbol for the data
exchange = 'binance'
symbol = 'BTC/USDT'

# Use ccxt to fetch the data from Binance
binance = ccxt.binance()
ohlcv = binance.fetch_ohlcv(symbol, '1d')

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.set_index('timestamp')

# Print the Bitcoin price in USD
print(df.head())

