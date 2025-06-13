import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import os


# Create pandas DataFrame with various financial institutions using yfinance.
#  
# Downloading data for American Express, Bank of America, Citibank, Capital One, Goldman Sachs, 
# Chase, Mastercard, US Bank, Visa, and Wells Fargo from June 1, 2024 to June 1, 2025.
# 
# Note: Only the daily adjusted close price column per each stock will be considered.
stocks = ['AXP','BAC','C','COF','GS','JPM','MA','USB','V','WFC']
df = yf.download(stocks, start='2024-06-01', end='2025-06-01', auto_adjust=False, progress=False)['Adj Close']

# Clean dataframe by dropping any columns that are completely empty and forward-filling missing data.
df = df.dropna(axis=1, how='all')
df = df.ffill()

# Preview dataframe.
print(df.head())

# Export dataframe of all imported stocks.
try:
    filename = "df.csv"
    df.to_csv(filename)

    # Get the full absolute path.
    full_path = os.path.abspath(filename)
    print(f"\nFile successfully saved to: {full_path}\n")

except Exception as e:
    print(f"\nAn error occurred while saving the file: {e}\n")


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

# Sort each column in ascending order.
#
# Function to, upon call, sort a list of data in ascending order, using selection sort for simplicity.
def sort(data):
    data = data.copy()
    L = length(data)
    for i in range(L):
        index = i
        for j in range(i + 1, L):
            if data[j] < data[index]:
                index = j
        data[i], data[index] = data[index], data[i]
    return data


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
    return sum_dev / (length(data) - 1)  # Note: Sample variance.

# Find standard deviation of each stock.
#  
# Function to find standard deviation of each stock.
def stddev(data):

    # Standard deviation is the square root of the variance.
    return variance(data) ** 0.5

# Find interquartile range of each stock.
# 
# Function to find interquartile range of each stock.
def iqr(data):
    data = sort(data)
    L = length(data)

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

    return interpolate(0.75) - interpolate(0.25)  # Return interquartile range, the 75th percentile minus the 25th percentile. 

# Combine statistics into csv.
#
# Create dataframe with columns as the tickers and the rows as the statistics.
stats = pd.DataFrame(columns=stocks, index=['Min','Max','Range','Mean','Variance','Std Dev', 'IQR'])

# Iterate through stocks and compute each statistic.
for ticker in stocks:
    data = df[ticker].tolist()

    stats[ticker] = [minimum(data), maximum(data), spread_range(data), mean(data), variance(data), stddev(data), iqr(data)]

# Preview statistics.
print(stats)

# Save statistics to file directory.
try:
    filename = "stats.csv"
    stats.to_csv(filename)

    # Get the full absolute path.
    full_path = os.path.abspath(filename)
    print(f"\nFile successfully saved to: {full_path}\n")

except Exception as e:
    print(f"\nAn error occurred while saving the file: {e}\n")


# Compute pearson correlation, spearman correlation, and kendall correlation. 
# Then, find which two stocks are most correlated with each other using each method.

# Find pearson correlation coefficient between each stock.
#  
# Function to find pearson correlation coefficient between each stock.
def pearson_corr(x, y):
    meanx = mean(x)
    meany = mean(y)
    covariancexy = 0
    variancex = 0
    variancey = 0

    # Formula for pearson correlation coefficient is covariance divided by the product of the standard deviations.
    for n in range(length(x)):
        covariancexy += (x[n] - meanx) * (y[n] - meany)
        variancex += (x[n] - meanx) ** 2
        variancey += (y[n] - meany) ** 2
    return covariancexy / ((variancex * variancey) ** 0.5)

# Function to compute and export correlation matrix and heatmap.
def get_matrix(df, stocks, corr, filename_prefix, title):
    matrix = pd.DataFrame(index=stocks, columns=stocks, dtype=float)
    for i in stocks:
        for j in stocks:
            if stocks.index(j) >= stocks.index(i):
                matrix.loc[i, j] = corr(df[i].tolist(), df[j].tolist())
            else:
                matrix.loc[i, j] = np.nan
    plt.figure(figsize=(10, 8))
    sb.heatmap(matrix, annot=True, cmap='Spectral')
    plt.title(f'{title} Correlation Heatmap')

    try:
        heatmap_filename = f"{filename_prefix}_heatmap.png"
        plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
        
        # Get the full absolute path.
        print(f"Heatmap image successfully saved to: {os.path.abspath(heatmap_filename)}\n")
    
    except Exception as e:
        print(f"An error occurred while saving the file: {e}\n")
    plt.close()

# Compute and save pearson correlation.
get_matrix(df, stocks, pearson_corr, "pearson", "Pearson")

# Find spearman correlation coefficient between each stock.
#  
# Function to find spearman correlation coefficient between each stock.
def spearman_corr(x, y):

    # Function to rank data, using average ranks for ties.
    def rank(data):
        L = length(data)
        data = [[data[i], i] for i in range(L)]
        data = sort(data)

        rank = [0] * L
        i = 0
        while i < L:
            val = data[i][0]
            j = i
            while j + 1 < L and data[j + 1][0] == val:
                j += 1
            avg_rank = (i + j + 2) / 2
            for k in range(i, j + 1):
                rank[data[k][1]] = avg_rank
            i = j + 1
        return rank

    # Rank the data and compute pearson correlation on the ranks.
    rankx = rank(x)
    ranky = rank(y)
    return pearson_corr(rankx, ranky)

# Compute and save spearman correlation.
get_matrix(df, stocks, spearman_corr, "spearman", "Spearman")

# Find kendall correlation coefficient between each stock.
#  
# Function to find kendall correlation coefficient between each stock.
def kendall_corr(x, y):
    L = length(x)
    concordant = 0
    discordant = 0

    # Formula for kendall correlation coefficient is the difference between the number of concordant and discordant pairs divided by the total number of pairs.
    for i in range(L):
        for j in range(i + 1, L):
            if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                concordant += 1
            elif (x[i] - x[j]) * (y[i] - y[j]) < 0:
                discordant += 1
    return (concordant - discordant) / ((L * (L - 1)) / 2)

# Compute and save kendall correlation.
get_matrix(df, stocks, kendall_corr, "kendall", "Kendall")

# Given all correlation methods, find the most correlated pairs of stocks.
#
# Function to find the most correlated pairs of credit network stocks, Visa, Mastercard, American Express, and Capital One.
stocks = ['V', 'MA', 'AXP', 'COF']  # Note: Discover (DFS) was acquired by Capital One (COF).
def greatest_corr(data, stocks, corr, method):
    L = length(stocks)
    max_corr = 0
    pair = None
    for i in range(L):
        for j in range(i + 1, L):
            corr_value = corr(data[stocks[i]].tolist(), data[stocks[j]].tolist())
            if corr_value > max_corr:
                max_corr = corr_value
                pair = (stocks[i], stocks[j])
    print(f"Most correlated pair ({method}): {pair} with correlation {max_corr}")
    return pair

# Return the most correlated pairs of stocks for each correlation method.
print("Best correlated pairs by method:")
pearson_pair = greatest_corr(df, stocks, pearson_corr, "Pearson")
spearman_pair = greatest_corr(df, stocks, spearman_corr, "Spearman")
kendall_pair = greatest_corr(df, stocks, kendall_corr, "Kendall")

# Set our new correlated pair to the spearman pair and perform cointegration on it.
pair = spearman_pair

# Given the two most correlated pairs of stocks, plot two subplots of the daily adjusted close price of each stock in the pair.
def plot_pair(data, pair):
    stocka, stockb = pair
    
    # Map tickers to company names.
    stock_names = {
        'AXP': 'American Express',
        'COF': 'Capital One',
        'MA': 'Mastercard',
        'V': 'Visa',
    }
    
    namea = stock_names[stocka] if stocka in stock_names else stocka
    nameb = stock_names[stockb] if stockb in stock_names else stockb
    
    # Plot the daily adjusted close price of each stock in the pair.
    # 
    # Stock A plots on the top.
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(data[stocka], label=namea, color='blue')
    plt.title(f'{namea} ({stocka}) Daily Adjusted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Stock B plots on the bottom.
    plt.subplot(2, 1, 2)
    plt.plot(data[stockb], label=nameb, color='red')
    plt.title(f'{nameb} ({stockb}) Daily Adjusted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Export the plot to a file.
    plt.tight_layout()

    try:
        plot_filename = f"{stocka}_{stockb}_price_plot.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        
        # Get the full absolute path.
        print(f"\nPlot image successfully saved to: {os.path.abspath(plot_filename)}")
    
    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")
    plt.close()
    
# Plot the most correlated pair based on Spearman correlation.
plot_pair(df, pair)

# Compute engle-granger cointegration.  *** TO BE COMPLETED
# 
# Normalize stock data on logarithmic scale.
# 
# Perform OLS regression.
# 
# Test residuals for stationarity.

# Utilize pairs trading methods to find optimal pairs trading strategy.  *** TO BE COMPLETED
#
# Concentrate signal with z-scores.
# 
# Backtest strategy.
# 
# Optimize threshholds.