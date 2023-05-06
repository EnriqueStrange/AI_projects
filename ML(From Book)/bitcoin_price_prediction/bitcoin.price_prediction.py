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
import ccxt
import pandas as pd

# Set the exchange and symbol for the data
exchange = 'binance'
symbol = 'BTC/USDT'

# Set the exchange and symbol for the data
exchange = 'binance'    # exchange where the data is fetched
symbol = 'BTC/USDT'    # trading pair symbol for Bitcoin in USDT

# Set the start and end date of the data
start_date = '2022-04-28'   # start date for the price data
end_date = '2023-04-28'     # end date for the price data

# Use ccxt to fetch the data from Binance
binance = ccxt.binance()    # create an instance of the Binance exchange class

# Fetch the OHLCV data for the desired time period
# OHLCV stands for Open, High, Low, Close, and Volume
# '1d' means daily time frame
# binance.parse8601() method is used to convert the start and end dates to timestamps
ohlcv = binance.fetch_ohlcv(symbol, '1m', binance.parse8601(start_date), binance.parse8601(end_date))

# Convert the data to a Pandas DataFrame
# ohlcv is a list of lists containing the OHLCV data for each day
# we convert it to a DataFrame with the columns: timestamp, Open, High, Low, Close, Volume
# we set the timestamp column as the index of the DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.set_index('timestamp')
 

# Print the Bitcoin price in USD
print(df.info())
