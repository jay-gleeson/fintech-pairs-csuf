import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


# Create pandas DataFrame with various financial institutions using yfinance.
#  
# Downloading data for Visa, Mastercard, Capital One, Chase, Citigroup, Wells Fargo,  
# American Express, Goldman Sachs, U.S. Bank, and Bank of America from June 1, 2024 to June 1, 2025.
# 
# Note: Only the daily adjusted close price column per each stock will be considered.
stocks = ['AXP','BAC','C','COF','GS','JPM','MA','USB','V','WFC']
df = yf.download(stocks, start='2024-06-01', end='2025-06-01', auto_adjust=False)['Adj Close']

# Export dataframe of all imported stocks.
df.to_csv("df.csv")


# Form a traditional correlation heatmap given current stocks as a proof of concept.
# 
# The purpose of this is to find what stocks are the most correlated 
# in order to pit the two most correlated ones against each other for pairs trading.
plt.figure(figsize=(10, 8))
sb.heatmap(df.corr(), annot=True, cmap='Spectral')
plt.title('Correlation Heatmap')

# Save correlation heatmap to file directory.
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')


# Using greedy methods, find min, max, mean, variance, standard deviation, 
# and interquartile range of stocks and download to csv for future use. 
#
# Find length of each column.
# 
# Function to find length of each column.
def length(data):
    count = 0
    for n in data:
        count += 1
    return count

# Find minimum and maximum of each stock.
#
# Function to find minimum daily adjusted close price.
def minimum(data):
    min = data[0]
    for n in data:
        if n < min:
            min = n
    return min

# Function to find maximum daily adjusted close price.
def maximum(data):
    max = data[0]
    for n in data:
        if n > max:
            max = n
    return max

# Find mean price of each stock.
# 
# Function to find mean daily adjusted close price of each stock.
def mean(data):
    sum = 0
    for n in data:
        sum += n
    return sum / length(data)

# Find variance of each stock.
# 
# Function to find variance of each stock.
def variance(data):
    sumsqdev = 0
    for n in data:
        sumsqdev += (n - mean(data)) ** 2
    return sumsqdev / (length(data) - 1)

# Find standard deviation of each stock.
#  
# Function to find standard deviation of each stock.
def stddev(data):
    return variance(data) ** 0.5

# Find interquartile range of each stock.  *** TO BE COMPLETED

# Combine statistics into csv.
#
# Create dataframe with columns as the tickers and the rows as the statistics.
stats = pd.DataFrame(columns=stocks, index=['Min','Max','Mean','Variance','Std Dev'])

# Iterate through stocks and compute each statistic.
for ticker in stocks:
    data = df[ticker].tolist()

    stats[ticker] = [minimum(data), maximum(data), mean(data), variance(data), stddev(data)]

# Save statistics to file directory.
print(stats)
stats.to_csv("stats.csv")


# Compute pearson correlation, spearman correlation, and kendall correlation.  *** TO BE COMPLETED

# Compute engle-granger cointegration.  *** TO BE COMPLETED

# Utilize pairs trading methods to find optimal pairs trading strategy.  *** TO BE COMPLETED