# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:20:03 2024

@author: WONCHAN
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

ticker = "^GSPC"
data = yf.download(ticker, start="2020-01-01", end="2023-12-31")
data['Return'] = data['Close'].pct_change()
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

X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (samples, look_back, 1)
y = torch.tensor(y, dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y = X.to(device), y.to(device)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])  
        return output

model = LSTMModel().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 500
batch_size = 32

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:  
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(X_train):.6f}")

model.eval()
with torch.no_grad():
    y_pred = model(X_test).cpu().numpy()
    y_test_actual = y_test.cpu().numpy()

y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test_actual.reshape(-1, 1))

excess_return = data['Return'].iloc[-len(y_test):].iloc[1:].values
predicted_return = np.diff(y_pred_rescaled.squeeze())
r2_excess_return = r2_score(excess_return, predicted_return)

print(f"S&P 500 초과수익률 R²: {r2_excess_return}")

plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(y_test):], y_test_rescaled, label="Actual Price")
plt.plot(data.index[-len(y_test):], y_pred_rescaled, label="Predicted Price")
plt.title(f"S&P 500 Index Prediction (PyTorch, GPU)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
