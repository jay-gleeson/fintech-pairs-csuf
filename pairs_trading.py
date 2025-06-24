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
stocks = ['AXP', 'BAC', 'C', 'COF', 'GS', 'JPM', 'MA', 'USB', 'V', 'WFC'] # Note: Later, only Visa (V), Mastercard (MA), American Express (AXP), and Capital One (COF) are considered.
df = yf.download(stocks, start='2024-06-01', end='2025-06-01', auto_adjust=False, progress=False)['Adj Close']

# Clean dataframe by dropping any columns that are completely empty and forward-filling missing data.
df = df.dropna(axis=1, how='all')
df = df.ffill()

# Preview dataframe.
print(df.head())

# Export dataframe of all imported stocks.
try:
    filename = 'df.csv'
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
# Function to, upon call, sort a series of data in ascending order, using selection sort for simplicity.
def sort(data):
    data = data.copy()
    L = length(data)
    for i in range(L):
        index = i
        for j in range(i + 1, L):
            if data.iloc[j] < data.iloc[index]:
                index = j
        data.iloc[i], data.iloc[index] = data.iloc[index], data.iloc[i]
    return data


# Find minimum and maximum of each stock.
#
# Function to find minimum daily adjusted close price.
def minimum(data):
    min = data.iloc[0]
    for n in data:
        if n < min:
            min = n
    return min

# Function to find maximum daily adjusted close price.
def maximum(data):
    max = data.iloc[0]
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
        
        # Function to find the minimum of two values.
        def minimum(a, b):
            if a < b:
                return a
            else:
                return b

        # Set high index to the next index.
        high = minimum(low + 1, L - 1)
        return data.iloc[low] + (data.iloc[high] - data.iloc[low]) * (pos - low)

    # Return interquartile range, the 75th percentile minus the 25th percentile.
    return interpolate(0.75) - interpolate(0.25)

# Combine statistics and print to console.
#
# Create dataframe with columns as the tickers and the rows as the statistics.
stats = pd.DataFrame(columns=stocks, index=['Min', 'Max', 'Range', 'Mean', 'Variance', 'Std Dev', 'IQR'])

# Iterate through stocks and compute each statistic.
for ticker in stocks:
    data = df[ticker]
    stats[ticker] = [minimum(data), maximum(data), spread_range(data), mean(data), variance(data), stddev(data), iqr(data)]

# Print statistics.
print(stats)

# Save statistics to file directory.
try:
    filename = 'stats.csv'
    stats.to_csv(filename)

    # Get the full absolute path.
    full_path = os.path.abspath(filename)
    print(f"\nFile successfully saved to: {full_path}\n")
except Exception as e:
    print(f"\nAn error occurred while saving the file: {e}\n")


# Compute pearson correlation, spearman correlation, and kendall correlation. 
# Then, find which two stocks are most correlated with each other using each method.
# 
# Function to compute and export correlation matrix and heatmap.
def get_matrix(data, stocks, corr, filename_prefix, title):
    matrix = pd.DataFrame(index=stocks, columns=stocks, dtype=float)
    for i in stocks:
        for j in stocks:
            if stocks.index(j) >= stocks.index(i):
                matrix.loc[i, j] = corr(data[i], data[j])
            else:
                matrix.loc[i, j] = np.nan
    plt.figure(figsize=(10, 8))
    sb.heatmap(matrix, annot=True, cmap='Spectral')
    plt.title(f'{title} Correlation Heatmap')

    # Save heatmap to file directory.
    try:
        heatmap_filename = f'{filename_prefix}_heatmap'
        matrix.to_csv(heatmap_filename + '.csv')
        plt.savefig(heatmap_filename + '.png', dpi=300, bbox_inches='tight')
        
        # Get the full absolute path.
        print(f"Heatmap image successfully saved to: {os.path.abspath(heatmap_filename + '.png')}\n")
        print(f"Heatmap CSV successfully saved to: {os.path.abspath(heatmap_filename + '.csv')}\n")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}\n")
    plt.close()

# Find pearson correlation coefficient between each stock.
#  
# Function to find pearson correlation coefficient between each stock.
def pearson_corr(x, y):
    L = length(x)
    meanx = mean(x)
    meany = mean(y)
    covariancexy = 0
    variancex = 0
    variancey = 0

    # Formula for pearson correlation coefficient is covariance divided by the product of the standard deviations.
    for n in range(L):
        covariancexy += (x.iloc[n] - meanx) * (y.iloc[n] - meany)
        variancex += (x.iloc[n] - meanx) ** 2
        variancey += (y.iloc[n] - meany) ** 2
    return covariancexy / ((variancex * variancey) ** 0.5)

# Compute and save pearson correlation.
get_matrix(df, stocks, pearson_corr, 'pearson', 'Pearson')

# Find spearman correlation coefficient between each stock.
#  
# Function to find spearman correlation coefficient between each stock.
def spearman_corr(x, y):

    # Function to rank data, using average ranks for ties.
    def rank(data):
        L = length(data)
        data = pd.Series([[data.iloc[n], n] for n in range(L)])
        data = sort(data)
        rank = [0] * L
        i = 0
        while i < L:
            val = data.iloc[i][0]
            j = i
            while j + 1 < L and data.iloc[j + 1][0] == val:
                j += 1
            avg_rank = (i + j + 2) / 2
            for k in range(i, j + 1):
                rank[data.iloc[k][1]] = avg_rank
            i = j + 1
        return pd.Series(rank, index=x.index)

    # Rank the data and compute pearson correlation on the ranks.
    rankx = rank(x)
    ranky = rank(y)
    return pearson_corr(rankx, ranky)

# Compute and save spearman correlation.
get_matrix(df, stocks, spearman_corr, 'spearman', 'Spearman')

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
            if (x.iloc[i] - x.iloc[j]) * (y.iloc[i] - y.iloc[j]) > 0:
                concordant += 1
            elif (x.iloc[i] - x.iloc[j]) * (y.iloc[i] - y.iloc[j]) < 0:
                discordant += 1
    return (concordant - discordant) / ((L * (L - 1)) / 2)

# Downsample the dataframe to speed up computation, taking 20% of the data.
#
# Function to downsample the dataframe by a given step.
def downsample_df(data, step):
    return data.iloc[::step].copy()

# Downsample the dataframe to speed up computation.
df_kendall = downsample_df(df, step=5)

# Compute and save kendall correlation.
get_matrix(df_kendall, stocks, kendall_corr, 'kendall', 'Kendall')

# Function to get the names of the stocks in the pair.
def get_stock_names(pair):
    stock_names = {
        'V': 'Visa',
        'MA': 'Mastercard',
        'AXP': 'American Express',
        'COF': 'Capital One'
    }
    namea = stock_names.get(pair[0], pair[0])
    nameb = stock_names.get(pair[1], pair[1])
    return namea, nameb

# Only consider stocks in new_stocks.
new_stocks = ['V', 'MA', 'AXP', 'COF'] # Note: Discover (DFS) was acquired by Capital One (COF).

# Given all correlation methods, find the most correlated pairs of stocks.
#
# Function to find the most correlated pairs of credit network stocks, Visa, Mastercard, American Express, and Capital One.
def greatest_corr(stocks, method):
    
    # Only consider stocks in new_stocks.
    relevant_stocks = [n for n in stocks if n in new_stocks]

    # Find the greatest correlation by reading the previously saved correlation CSV files.
    corr_csv = f'{method.lower()}_heatmap.csv'
    pair = None
    max_corr = None
    if os.path.exists(corr_csv):
        corr_df = pd.read_csv(corr_csv, index_col=0)
        for i in relevant_stocks:
            for j in relevant_stocks:
                if i != j:
                    val = float(corr_df.loc[i, j])
                    if (max_corr is None) or (val > max_corr):
                        max_corr = val
                        pair = (i, j)
        if pair:

            # Put the stock with the lower average daily adjusted close price first in the pair for consistency.
            if os.path.exists('stats.csv'):
                stats_df = pd.read_csv('stats.csv', index_col=0)
                means = {n: float(stats_df.loc['Mean', n]) for n in pair}
                if means[pair[0]] > means[pair[1]]:
                    pair = (pair[1], pair[0])

                # Get the names of the stocks in the pair.
                namea, nameb = get_stock_names(pair)

                # Print the most correlated pair with their names and correlation value.
                print(f"Most correlated pair ({method}): {namea} ({pair[0]}) and {nameb} ({pair[1]}) with correlation {max_corr}")
            else:
                print(f"Stats CSV file not found. Defaulting to original order: {pair} with correlation {max_corr}")
    else:
        print(f"Correlation CSV file '{corr_csv}' not found.")
    return pair

# Return the most correlated pairs of stocks for each correlation method.
print("\nBest correlated pairs by method:")
pearson_pair = greatest_corr(stocks, 'Pearson')
spearman_pair = greatest_corr(stocks, 'Spearman')
kendall_pair = greatest_corr(stocks, 'Kendall')

# Set pair to be considered for pairs trading as the pair with the highest spearman correlation value.
# 
# Function to set and output new pair given spearman correlation highest valued pair.
def set_pair(spearman_pair):

    # Get the names of the stocks in the pair.
    stocka, stockb = spearman_pair
    namea, nameb = get_stock_names(spearman_pair)
    print(f"\nPair to be considered for pairs trading: {namea} ({stocka}) and {nameb} ({stockb})")
    return spearman_pair

# Set pair as spearman pair and display result to output.
pair = set_pair(spearman_pair)

# Given the two most correlated pairs of stocks, plot two subplots of the daily adjusted close price of each stock in the pair.
def plot_pair(data, pair):

    # Get the names of the stocks in the pair.
    stocka, stockb = pair
    namea, nameb = get_stock_names(pair)
    
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
    
    # Save plot to file directory.
    try:
        plot_filename = f'{stocka}_{stockb}_price_plot.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        
        # Get the full absolute path.
        print(f"\nPlot image successfully saved to: {os.path.abspath(plot_filename)}")
    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")
    plt.close()
    
# Plot the most correlated pair based on Spearman correlation.
plot_pair(df, pair)


# Compute engle-granger cointegration.
# 
# Function to normalize stock data on logarithmic scale.
def normalize(data, pair):
    stocka, stockb = pair
    a = data[stocka]
    b = data[stockb]
    norma = np.log(a)
    normb = np.log(b)
    return norma, normb

# Normalize the pair of stocks.
norma, normb = normalize(df, pair)

# Function to perform OLS regression on the normalized pair of stocks.
def ols(a, b):
    L = length(a)
    meana = mean(a)
    meanb = mean(b)
    num = 0
    den = 0

    # Formula for OLS regression slope and intercept where slope is the ratio of covariance to variance and intercept is the mean of b minus the slope times the mean of a.
    for n in range(L):
        num += (a.iloc[n] - meana) * (b.iloc[n] - meanb)
        den += (a.iloc[n] - meana) ** 2
    slope = num / den
    intercept = meanb - slope * meana
    residuals = pd.Series([b.iloc[n] - (slope * a.iloc[n] + intercept) for n in range(L)], index=a.index)
    residuals_lag = residuals.shift(1)
    residuals_diff = residuals - residuals_lag
    return residuals, residuals_lag, residuals_diff

# Perform OLS regression on the normalized pair of stocks.
residuals, residuals_lag, residuals_diff = ols(norma, normb)

# Generate OLS best fit line plot for the normalized pair (norma vs normb).
# 
# Function to plot OLS best fit line plot.
def plot_ols(pair):
    L = length(norma)
    meana = mean(norma)
    meanb = mean(normb)
    num = 0
    den = 0
    for n in range(L):
        num += (norma.iloc[n] - meana) * (normb.iloc[n] - meanb)
        den += (norma.iloc[n] - meana) ** 2
    slope = num / den
    intercept = meanb - slope * meana

    # Get stock names.
    stocka, stockb = pair
    namea, nameb = get_stock_names(pair)
    
    # Plot the data and the best fit line.
    plt.figure(figsize=(10, 6))
    plt.scatter(norma, normb, label='Data Points', alpha=0.6)
    plt.plot(norma, slope * norma + intercept, color='red', label='OLS Best Fit Line')
    plt.xlabel(f'{namea} ({stocka})')
    plt.ylabel(f'{nameb} ({stockb})')
    plt.title(f'OLS Best Fit Line: {namea} ({stocka}) vs {nameb} ({stockb})')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Save plot to file directory.
    try:
        ols_best_fit_filename = f'{stocka}_{stockb}_best_fit_plot.png'
        plt.savefig(ols_best_fit_filename, dpi=300, bbox_inches='tight')
        
        # Get the full absolute path.
        print(f"\nOLS best fit plot image successfully saved to: {os.path.abspath(ols_best_fit_filename)}")
    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")
    plt.close()

# Plot pair.
plot_ols(pair)

# Function to plot residuals.
def plot_residuals(residuals, pair, data):

    # Plot residuals to visualize the relationship between the two stocks.
    stocka, stockb = pair
    namea, nameb = get_stock_names(pair)
    plt.figure(figsize=(10, 6))

    # Enter residuals into dataframe, to display dates on plot.
    df_residuals = pd.Series(residuals, index=data.index)
    plt.plot(df_residuals, label='Residuals', color='purple')
    plt.title(f'Residuals of {namea} and {nameb}')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Save plot to file directory.
    try:
        residual_plot_filename = f'{stocka}_{stockb}_residuals_plot.png'
        plt.savefig(residual_plot_filename, dpi=300, bbox_inches='tight')
        
        # Get the full absolute path.
        print(f"\nResidual plot image successfully saved to: {os.path.abspath(residual_plot_filename)}")
    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")
    plt.close()

# Plot the residuals of the normalized pair of stocks.
plot_residuals(residuals, pair, df)

# Test residuals for stationarity.
# 
# Enter and clean residuals into a dataframe.
df_residuals = pd.DataFrame({'residual': residuals, 'residual_lag': residuals_lag, 'residual_diff': residuals_diff}).dropna().reset_index(drop=True)

# Basic ADF test with no lag, no constant, and no trend.
def basic_adfuller(df):
    X = df['residual_lag'].to_numpy().reshape(-1, 1)
    Y = df['residual_diff'].to_numpy().reshape(-1, 1)
    XtX = X.T @ X
    XtY = X.T @ Y
    gamma_hat = XtY / XtX
    y_pred = X * gamma_hat
    errors = Y - y_pred
    RSS = 0
    for err in errors:
        RSS += err ** 2
    n = 0
    for i in X:
        n += 1
    sigma2 = RSS / (n - 1)
    sum_xsq = 0
    for x in X:
        sum_xsq += x ** 2
    var_gamma = sigma2 / sum_xsq
    se_gamma = var_gamma ** 0.5
    gamma_hat_scalar = gamma_hat[0, 0]
    t_stat = gamma_hat_scalar / se_gamma
    print("\nBasic ADF T-statistic:", t_stat.item())

    # Approximate p-value using an approximation error function. 
    def erf(x):
        sign = 1 if x >= 0 else -1
        x = abs(x)
        t = 1.0 / (1.0 + 0.3275911 * x)
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * np.exp(-x * x)
        return sign * y
    p_value = 1 - erf(abs(t_stat) / np.sqrt(2))
    print("Basic ADF p-value:", p_value.item())

# Calculate ADF.
basic_adfuller(df_residuals)

# Utilize pairs trading methods to find optimal pairs trading strategy.
# 
# Create a new dataframe containing only the columns for the selected pair.
df_trading = df[list(pair)].copy()

# Function to plot the spread between the two stocks in the pair.
def plot_spread(data, pair):

    # Calculate and plot the spread between the two stocks in the pair.
    data['spread'] = data[pair[1]] - data[pair[0]]

    # Get the names of the stocks in the pair.
    stocka, stockb = pair
    namea, nameb = get_stock_names(pair)

    # Plot spread.
    plt.figure(figsize=(10, 6))
    plt.plot(data['spread'], label=f'Spread: {nameb} - {namea}', color='green')
    plt.title(f'Spread between {nameb} and {namea}')
    plt.xlabel('Date')
    plt.ylabel('Spread')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Save plot to file directory.
    try:
        spread_plot_filename = f'{stocka}_{stockb}_spread_plot.png'
        plt.savefig(spread_plot_filename, dpi=300, bbox_inches='tight')
        
        # Get the full absolute path.
        print(f"\nSpread plot image successfully saved to: {os.path.abspath(spread_plot_filename)}")
    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")
    plt.close()

# Plot the spread.
plot_spread(df_trading, pair)

# Function to plot the z-score of the spread and generate trading signals.
def plot_zscore(data):
    
    # Define z-score to normalize the spread.
    data['zscore'] = (data['spread'] - mean(data['spread'])) / stddev(data['spread'])

    # Set thresholds for entering trades.
    upper_threshold = 1
    lower_threshold = -1

    # Initialize position column.
    data['pos'] = 0
    position = 0

    for n in range(length(data['zscore'])):
        z = data['zscore'].iloc[n]
        if position == 0:
            if z > upper_threshold:
                position = -1
            elif z < lower_threshold:
                position = 1
        elif position == 1 and z > 0:
            position = 0
        elif position == -1 and z < 0:
            position = 0
        data.iloc[n, data.columns.get_loc('pos')] = position

    # Plot z-score.
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['zscore'], label='Z-Score')
    plt.axhline(upper_threshold, color='red', linestyle='--', label='Upper Threshold')
    plt.axhline(lower_threshold, color='green', linestyle='--', label='Lower Threshold')
    plt.title('Z-Score of the Spread with Trade Signals')
    plt.xlabel('Date')
    plt.ylabel('Z-Score')
    plt.legend()

    # Save plot to file directory.
    try:
        zscore_plot_filename = f'{pair[0]}_{pair[1]}_zscore_plot.png'
        plt.savefig(zscore_plot_filename, dpi=300, bbox_inches='tight')
        
        # Get the full absolute path.
        print(f"\nZ-Score plot image successfully saved to: {os.path.abspath(zscore_plot_filename)}")
    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")
    plt.close()

# Plot the z-score of the spread.
plot_zscore(df_trading)

# Function to plot the cumulative returns of the pairs trading strategy.
def plot_cumulative_returns(data, pair):
    
    # Compute daily returns for each stock in the pair.
    data[f'{pair[1]}_return'] = (data[pair[1]] - data[pair[1]].shift(1)) / data[pair[1]].shift(1)
    data[f'{pair[0]}_return'] = (data[pair[0]] - data[pair[0]].shift(1)) / data[pair[0]].shift(1)
    
    # Add buy/sell for each stock in df to plot later.
    buy_col_0 = f'buy_{pair[0]}'
    buy_col_1 = f'buy_{pair[1]}'
    sell_col_0 = f'sell_{pair[0]}'
    sell_col_1 = f'sell_{pair[1]}'

    # Ensure columns exist and are initialized to 0.0 if not already present.
    for col in [buy_col_0, buy_col_1, sell_col_0, sell_col_1]:
        if col not in data.columns:
            data[col] = 0.0
        else:
            data[col].values[:] = 0.0
    
    # Mark buy/sell signals based on position changes.
    for n in range(1, length(data['spread'])):
        pos = data['pos'].iloc[n]
        prev_pos = data['pos'].iloc[n-1]

        # Enter long: pos changes from 0 to 1.
        if prev_pos == 0 and pos == 1:
            data.at[data.index[n], buy_col_0] = data.at[data.index[n], pair[0]]
            data.at[data.index[n], sell_col_1] = data.at[data.index[n], pair[1]]
        # Enter short: pos changes from 0 to -1.
        elif prev_pos == 0 and pos == -1:
            data.at[data.index[n], sell_col_0] = data.at[data.index[n], pair[0]]
            data.at[data.index[n], buy_col_1] = data.at[data.index[n], pair[1]]
        # Exit long: pos changes from 1 to 0.
        elif prev_pos == 1 and pos == 0:
            data.at[data.index[n], sell_col_0] = data.at[data.index[n], pair[0]]
            data.at[data.index[n], buy_col_1] = data.at[data.index[n], pair[1]]
        # Exit short: pos changes from -1 to 0.
        elif prev_pos == -1 and pos == 0:
            data.at[data.index[n], buy_col_0] = data.at[data.index[n], pair[0]]
            data.at[data.index[n], sell_col_1] = data.at[data.index[n], pair[1]]

    # Strategy returns.
    data['strategy_return'] = data['pos'].shift(1) * (data[f'{pair[1]}_return'] - data[f'{pair[0]}_return'])

    # Cumulative returns.
    cumulative_return = []
    prod = 1
    for n in data['strategy_return'].fillna(0):
        prod *= (1 + n)
        cumulative_return.append(prod)
    data['cumulative_return'] = cumulative_return

    # Plot cumulative returns.
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['cumulative_return'], label='Cumulative Return from Strategy')
    plt.title('Cumulative Returns of Pairs Trading Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')

    # Save plot to file directory.
    try:
        cumret_plot_filename = f'{pair[0]}_{pair[1]}_cumulative_returns_plot.png'
        plt.savefig(cumret_plot_filename, dpi=300, bbox_inches='tight')
        
        # Get the full absolute path.
        print(f"\nCumulative returns plot image successfully saved to: {os.path.abspath(cumret_plot_filename)}")
    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")
    plt.close()

# Plot the cumulative returns of the pairs trading strategy.
plot_cumulative_returns(df_trading, pair)

# Find overall profit percentage.
# 
# Function to return profit percentage from cumulative returns.
def find_profit(data):
    initial = data['cumulative_return'].iloc[0]
    final = data['cumulative_return'].iloc[-1]
    profit_pct = (final - initial) / initial * 100
    print(f"\nFinal profit percentage: {profit_pct:.4f}%")

# Return profit percentage from cumulative returns.
find_profit(df_trading)

# Function to return the sharpe ratio and max drawdown of the pairs trading strategy.
def sharpe_maxdrawdown(data):
    
    # Calculate sharpe ratio.
    count = 0
    total = 0
    for n in data['strategy_return']:
        if not pd.isna(n):
            total += n
            count += 1
    strategy_return_mean = total / count
    sum_sq = 0
    mean_val = strategy_return_mean
    count = 0
    for n in data['strategy_return']:
        if not pd.isna(n):
            sum_sq += (n - mean_val) ** 2
            count += 1
    strategy_return_std = (sum_sq / (count - 1)) ** 0.5
    sharpe_ratio = strategy_return_mean / strategy_return_std * (length(data) ** 0.5)
    print(f"\nSharpe Ratio: {sharpe_ratio}")

    # Calculate max drawdown.
    cumulative_max = []
    current_max = data['cumulative_return'].iloc[0]
    for n in data['cumulative_return']:
        if n > current_max:
            current_max = n
        cumulative_max.append(current_max)
    cumulative_max = pd.Series(cumulative_max, index=data.index)
    drawdown = (cumulative_max - data['cumulative_return']) / cumulative_max
    max_drawdown = 0
    for n in drawdown:
        if n > max_drawdown:
            max_drawdown = n
    print(f"Max Drawdown: {max_drawdown}")

# Return the sharpe ratio and max drawdown of the pairs trading strategy.
sharpe_maxdrawdown(df_trading)

def plot_pair_trades(data, pair):

    # Get stock names.
    stocka, stockb = pair
    namea, nameb = get_stock_names(pair)

    # Top subplot: Stock A.
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data[stocka], label=namea, color='blue', zorder=1)
    buy_a = data[data[f'buy_{stocka}'] > 0]
    sell_a = data[data[f'sell_{stocka}'] > 0]
    plt.scatter(buy_a.index, buy_a[stocka], marker='^', color='green', edgecolors='black', label='Buy', s=100, zorder=2)
    plt.scatter(sell_a.index, sell_a[stocka], marker='v', color='red', edgecolors='black', label='Sell', s=100, zorder=2)
    plt.title(f'{namea} ({stocka}) Pair Trade History')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Bottom subplot: Stock B.
    plt.subplot(2, 1, 2)
    plt.plot(data.index, data[stockb], label=nameb, color='red', zorder=1)
    buy_b = data[data[f'buy_{stockb}'] > 0]
    sell_b = data[data[f'sell_{stockb}'] > 0]
    plt.scatter(buy_b.index, buy_b[stockb], marker='^', color='green', edgecolors='black', label='Buy', s=100, zorder=2)
    plt.scatter(sell_b.index, sell_b[stockb], marker='v', color='red', edgecolors='black', label='Sell', s=100, zorder=2)
    plt.title(f'{nameb} ({stockb}) Pair Trade History')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    
    # Save plot to file directory.
    try:
        plot_pair_filename = f'{pair[0]}_{pair[1]}_plot_pair.png'
        plt.savefig(plot_pair_filename, dpi=300, bbox_inches='tight')
        
        # Get the full absolute path.
        print(f"\nPlot pair image successfully saved to: {os.path.abspath(plot_pair_filename)}")
    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")
    plt.close()

# Plot the pair trades.
plot_pair_trades(df_trading, pair)
