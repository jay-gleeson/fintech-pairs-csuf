import yfinance as yf
import pandas as pd
import numpy as np

# Create pandas DataFrame with various financial institutions using yfinance.
# Downloading data for Visa, Mastercard, Discover Financial Services, JPMorgan Chase, Citigroup, Wells Fargo, American Express, Goldman Sachs, U.S. Bancorp, and Bank of America from June 1, 2024 to June 1, 2025.
df = yf.download(['V', 'MA', 'DFS', 'JPM', 'C', 'WFC', 'AXP', 'GS', 'USB', 'BAC'], start='2024-06-01', end='2025-06-01')

print(df.head())

#test commit