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

# Find range of each stock.
# 
# Function to find the range of each stock.
def spread_range(data):
    return maximum(data) - minimum(data)

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
    m = mean(data)
    sum_dev = 0

    # Numerator of variance formula, find sum of squared deviations.
    for n in data:
        sum_dev += (n - m) ** 2
    return sum_dev / (length(data) - 1) #  Sample variance.

# Find standard deviation of each stock.
#  
# Function to find standard deviation of each stock.
def stddev(data):
    return variance(data) ** 0.5  # Square root of sample variance.

# Find interquartile range of each stock.
# 
# Function to find interquartile range of each stock.
def iqr(data):
    data = data.copy()
    L = length(data)

    # Sort prices in ascending order, using selection sort for simplicity.
    for i in range(L):
        index = i
        for j in range(i + 1, L):
            if data[j] < data[index]:
                index = j
        data[i], data[index] = data[index], data[i]

    # Interpolation function to find quartiles, using linear interpolation between adjacent sorted values. 
    def interpolate(quartile):
        pos = quartile * (L - 1)
        low = int(pos)
        
        def minimum(a, b):
            if a < b:
                return a
            else:
                return b
    
        high = minimum(low + 1, L - 1)
        return data[low] + (data[high] - data[low]) * (pos - low)

    return interpolate(0.75) - interpolate(0.25)  # Return inter-quartile range, which is the 75th percentile minus the 25th percentile. 

# Combine statistics into csv.
#
# Create dataframe with columns as the tickers and the rows as the statistics.
stats = pd.DataFrame(columns=stocks, index=['Min','Max','Range','Mean','Variance','Std Dev', 'IQR'])

# Iterate through stocks and compute each statistic.
for ticker in stocks:
    data = df[ticker].tolist()

    stats[ticker] = [minimum(data), maximum(data), spread_range(data), mean(data), variance(data), stddev(data), iqr(data)]

# Save statistics to file directory.
print(stats)
stats.to_csv("stats.csv")


# Compute pearson correlation, spearman correlation, and kendall correlation.  *** TO BE COMPLETED

# Compute engle-granger cointegration.  *** TO BE COMPLETED

# Utilize pairs trading methods to find optimal pairs trading strategy.  *** TO BE COMPLETED
