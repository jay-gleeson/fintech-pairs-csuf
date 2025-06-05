import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# Create pandas DataFrame with various financial institutions using yfinance.
# 
# Downloading data for Visa, Mastercard, Capital One, Chase, Citigroup, Wells Fargo, 
# American Express, Goldman Sachs, U.S. Bank, and Bank of America from June 1, 2024 to June 1, 2025.
df = yf.download(['V', 'MA', 'COF', 'JPM', 'C', 'WFC', 'AXP', 'GS', 'USB', 'BAC'], start='2024-06-01', end='2025-06-01', auto_adjust = False)

# Drop all columns except for the adj close price.
df.drop(columns = ['Open', 'High', 'Low','Close', 'Volume'], axis = 1, inplace = True)

# Drop the top or title column of the dataframe.
df = df.droplevel(0, axis = 1)

# Detail the most important figure information per each stock, including mean, standard deviation, min, and max.
info = df.describe()
info.to_csv("info.csv")

# Form a traditional correlation heatmap given current stocks as a proof of concept.
# 
# The purpose of this is to find what stocks are the most correlated 
# in order to pit the two most correlated ones against each other for pairs trading.
plt.figure(figsize=(10, 8))
sb.heatmap(df.corr(), annot=True, cmap='Spectral')
plt.title('Correlation Heatmap')
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
