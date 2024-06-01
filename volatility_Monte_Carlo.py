# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 20:34:35 2024

@author: WONCHAN
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# 주식 데이터를 다운로드합니다.
tickers = ['000220.KS', '005930.KS', '004770.KS']
stock_data = yf.download(tickers, start='2020-01-01', end='2024-06-01')['Adj Close']

# 일별 수익률을 계산합니다.
daily_returns = stock_data.pct_change().dropna()

# 시뮬레이션 파라미터 설정
num_portfolios = 10000  # 생성할 랜덤 포트폴리오의 수
results = np.zeros((4, num_portfolios))  # 수익률, 변동성, 샤프 비율, 각 주식 가중치

np.random.seed(42)

# 랜덤 포트폴리오 생성 및 평가
for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)  # 가중치의 합을 1로 맞춥니다.
    
    # 포트폴리오 수익률 및 변동성 계산
    portfolio_return = np.sum(weights * daily_returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    # 결과 저장
    results[0,i] = portfolio_return
    results[1,i] = portfolio_volatility
    results[2,i] = sharpe_ratio
    results[3,i] = weights[0]

# 결과를 데이터프레임으로 변환
results_frame = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe Ratio', 'Weight 1'])

# 최적의 포트폴리오 찾기 (샤프 비율이 가장 높은 포트폴리오)
max_sharpe_idx = results_frame['Sharpe Ratio'].idxmax()
max_sharpe_portfolio = results_frame.loc[max_sharpe_idx]

print(f"Optimal Weights: {weights}")
print(f"Expected Return: {max_sharpe_portfolio['Return']}")
print(f"Expected Volatility: {max_sharpe_portfolio['Volatility']}")
print(f"Sharpe Ratio: {max_sharpe_portfolio['Sharpe Ratio']}")

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(results_frame['Volatility'], results_frame['Return'], c=results_frame['Sharpe Ratio'], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(max_sharpe_portfolio[1], max_sharpe_portfolio[0], c='red', marker='*', s=200)
plt.title('Efficient Frontier')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.show()
