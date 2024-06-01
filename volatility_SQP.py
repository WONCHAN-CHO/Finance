# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 20:06:49 2024

@author: WONCHAN
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize


tickers = ['000220.KS', '005930.KS', '004770.KS']
stock_data = yf.download(tickers, start='2020-01-01', end='2024-06-01')['Adj Close']


daily_returns = stock_data.pct_change().dropna()


def calc_portfolio_volatility(weights, returns):
    cov_matrix = returns.cov()
    var_portfolio = np.dot(weights.T, np.dot(cov_matrix, weights))
    vol_portfolio = np.sqrt(var_portfolio) * np.sqrt(252)
    return vol_portfolio


initial_weights = np.array([1/len(tickers)] * len(tickers))


constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 1) for _ in range(len(tickers)))


optimization_result = minimize(calc_portfolio_volatility, initial_weights, args=(daily_returns,), 
                               method='SLSQP', bounds=bounds, constraints=constraints)


optimal_weights = optimization_result.x
min_volatility = calc_portfolio_volatility(optimal_weights, daily_returns)

print(f"Optimal Weights: {optimal_weights}")
print(f"Minimum Volatility: {min_volatility}")


plt.figure(figsize=(10, 6))
plt.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', startangle=140)
plt.title('Optimal Portfolio Weights')
plt.show()


fig, ax = plt.subplots(figsize=(10, 6))
cumulative_returns = (daily_returns @ optimal_weights).cumsum()
cumulative_returns.index = daily_returns.index  
ax.plot(cumulative_returns.index, cumulative_returns, label='Optimal Portfolio')
ax.set_title('Cumulative Returns of Optimal Portfolio')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Returns')
ax.legend()
plt.show()


annual_volatility = daily_returns.std() * np.sqrt(252)
annual_volatility.plot(kind='bar', figsize=(10, 6), title='Annual Volatility of Individual Stocks')
plt.ylabel('Volatility')
plt.show()




