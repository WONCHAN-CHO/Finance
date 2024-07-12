# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:17:31 2024

@author: WONCHAN
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

ticker = '^KS200'
data = yf.download(ticker, start='2000-01-01', end='2024-12-31')

data['Returns'] = data['Adj Close'].pct_change()
data.dropna(inplace=True)

data['Volatility'] = data['Returns'].rolling(window=30).std() * np.sqrt(252)
data.dropna(inplace=True)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Volatility']])

train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 30
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, batch_size=1, epochs=1)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Volatility'], label='Actual Volatility')
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict)+time_step, :] = train_predict

test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(time_step*2)+1:len(data)-1, :] = test_predict

plt.plot(data.index, train_predict_plot, label='Train Predict')
plt.plot(data.index, test_predict_plot, label='Test Predict')
plt.title('KOSPI 200 Volatility Prediction')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.show()

future_steps = 30  
n_simulations = 100  

all_future_predicts = []

for _ in range(n_simulations):
    future_predict = []
    last_sequence = scaled_data[-time_step:]

    for _ in range(future_steps):
        last_sequence = last_sequence.reshape((1, time_step, 1))
        next_volatility = model.predict(last_sequence)
        future_predict.append(next_volatility[0, 0])
        last_sequence = np.append(last_sequence[:, 1:, :], np.array(next_volatility).reshape(1, 1, 1), axis=1)
    
    future_predict = scaler.inverse_transform(np.array(future_predict).reshape(-1, 1))
    all_future_predicts.append(future_predict.flatten())

future_predict_mean = np.mean(all_future_predicts, axis=0)

future_dates = pd.date_range(start=data.index[-1], periods=future_steps+1, inclusive='right')

plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Volatility'], label='Actual Volatility')
plt.plot(future_dates, future_predict_mean, label='Future Predict (Mean)', linestyle='dashed')
plt.title('KOSPI 200 Future Volatility Prediction')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.show()
