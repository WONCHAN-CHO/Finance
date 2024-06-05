# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:11:39 2024

@author: WONCHAN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load historical data
data = pd.read_csv(’historical_data.csv’)
S = data[’Close’].values
r = data[’RiskFreeRate’].values
sigma = data[’Volatility’].values
T = len(S)
K = 100 # Strike price
dt = 1 / 252 # Time step (daily)
# Black-Scholes d1 function
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
plt.plot(portfolio_values, label=’Hedged Portfolio’)
plt.plot(S, label=’Underlying Asset’)
plt.legend()
plt.show()