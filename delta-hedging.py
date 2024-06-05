# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:15:27 2024

@author: WONCHAN
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm  

file_path = 'C:/Users/WONCHAN/kospi200_with_risk_free_rate_and_volatility.csv'

if not os.path.isfile(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# Load historical data
data = pd.read_csv(file_path)

# Print the column names to check if they are correct
print("Column names in the CSV file:", data.columns)

# Assuming the 'RiskFreeRate' is missing, let's add a constant risk-free rate of 2% per annum
if 'RiskFreeRate' not in data.columns:
    data['RiskFreeRate'] = 0.02 / 252  # Converting annual rate to daily rate

# Check if 'Volatility' is present, if not, we might need to calculate or assume it
if 'Volatility' not in data.columns:
    raise KeyError("The required column 'Volatility' is not found in the CSV file.")

S = data['Close'].values
r = data['RiskFreeRate'].values
sigma = data['Volatility'].values
T = len(S)

K = 100  # Strike price
dt = 1 / 252  # Time step (daily)

# Black-Scholes
def d1(S, K, r, sigma, T):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

# Delta function
def delta(S, K, r, sigma, T):
    return norm.cdf(d1(S, K, r, sigma, T))

# Initialize portfolio
portfolio_values = []
portfolio = 0
previous_delta = 0

# Backtesting loop
for t in range(T):
    current_delta = delta(S[t], K, r[t], sigma[t], T - t * dt)
    portfolio += (current_delta - previous_delta) * S[t]
    portfolio_values.append(portfolio)
    previous_delta = current_delta

# Calculate performance metrics
portfolio_values = np.array(portfolio_values)
hedging_error = S[-1] - portfolio_values[-1]
variance_reduction = np.var(S) - np.var(portfolio_values)
cost_of_hedging = np.sum(np.abs(np.diff(portfolio_values)))

print(f"Hedging Error: {hedging_error:.2f}")
print(f"Variance Reduction: {variance_reduction:.2f}")
print(f"Cost of Hedging: {cost_of_hedging:.2f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(portfolio_values, label='Hedged Portfolio')
plt.plot(S, label='Underlying Asset')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.title('Comparison of Hedged Portfolio and Underlying Asset')
plt.savefig('C:/Users/chln0/Desktop/hedging_results.png')  
plt.show()
