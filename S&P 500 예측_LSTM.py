# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:02:20 2024

@author: WONCHAN
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

ticker = "^GSPC"
data = yf.download(ticker, start="2020-01-01", end="2023-12-31")
data['Return'] = data['Close'].pct_change()
data.dropna(inplace=True)

ticker_irx = "^IRX"
irx_data = yf.download(ticker_irx, start="2020-01-01", end="2023-12-31")
irx_data['RiskFreeRate'] = irx_data['Close'] / 100

data = data.join(irx_data['RiskFreeRate'], how='inner')

data['ExcessReturn'] = data['Return'] - data['RiskFreeRate'].shift(1)
data.dropna(inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 60
X, y = create_dataset(scaled_data, look_back)
X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, batch_size=32, epochs=500, verbose=1)

predicted_scaled = model.predict(X_test)
predicted = scaler.inverse_transform(predicted_scaled)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

excess_return = data['ExcessReturn'].iloc[-len(y_test):]  
predicted_return = np.diff(predicted.squeeze()) 

excess_return = excess_return.iloc[1:].values

r2_excess_return = r2_score(excess_return, predicted_return)
print(f"S&P 500 초과수익률 R²: {r2_excess_return}")

plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(y_test):], y_test_actual, label="Actual Price")
plt.plot(data.index[-len(y_test):], predicted, label="Predicted Price")
plt.title(f"S&P 500 Index Prediction (2020-2023)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()