# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:47:12 2024

@author: WONCHAN
"""

import yfinance as yf
import pandas as pd
import numpy as np

# KOSPI200 데이터 다운로드
ticker = "^KS200"
data = yf.download(ticker, start="2014-01-01", end="2024-06-05")

# 종가 데이터 사용
data = data[['Close']]

# 리스크 프리 레이트 추가 (연 2%를 일간 비율로 변환)
data['RiskFreeRate'] = 0.02 / 252

# 변동성 계산 (일간 수익률의 표준 편차)
data['Returns'] = data['Close'].pct_change()
data['Volatility'] = data['Returns'].rolling(window=21).std() * np.sqrt(252)  # 21일 이동 표준편차 사용

# NaN 값 제거
data.dropna(inplace=True)

# 필요한 컬럼만 선택
data = data[['Close', 'RiskFreeRate', 'Volatility']]

# 준비된 데이터 확인
print(data.head())

# 준비된 데이터를 CSV 파일로 저장
file_path = 'kospi200_with_risk_free_rate_and_volatility.csv'
data.to_csv(file_path, index=True)
print(f"Data saved to {file_path}")



