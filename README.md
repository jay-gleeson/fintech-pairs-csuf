<div align="center">
  <img alt="GitHub License" src="https://img.shields.io/github/license/jay-gleeson/fintech-pairs-csuf">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&amp;logoColor=fff" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter_Notebook-F37626?logo=jupyter&amp;logoColor=fff" alt="Jupyter Notebook">
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&amp;logoColor=fff" alt="Google Colab">
  <img src="https://img.shields.io/badge/-yfinance-410093?logo=yfinance&amp;logoColor=fff" alt="yfinance">
  <img src="https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&amp;logoColor=fff" alt="Matplotlib">
  <img src="https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&amp;logoColor=fff" alt="NumPy">
  <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&amp;logoColor=fff" alt="Pandas">
  <img src="https://img.shields.io/badge/-Seaborn-6BA1AE?logo=seaborn&amp;logoColor=fff" alt="Seaborn">
</div>

# Visa (V) versus Mastercard (MA) Pairs Trading 💳
Pairs trading project to find correlation, cointegration, and perform pairs trading methods on credit networks Visa and Mastercard. 🏦

Project originally completed by [Jay Gleeson](https://github.com/jay-gleeson/) and [Reyli Hernandez](https://github.com/rey-hern). This project was conducted with the help of the [CIC | PCUBED](https://www.fullerton.edu/ecs/cicpcubed/) organization at California State University, Fullerton.

## Background 🗣️
   Visa (V) and Mastercard (MA) are two dominant players in the global payments industry, operating as credit card networks that facilitate cashless transactions. Unlike issuers, these companies serve solely as payment networks, forming partnerships with numerous banks while remaining independent from any single issuer. This business model not only sets them apart from competitors like American Express and Discover ([recently acquired by Capital One Financial Corp](https://www.businesswire.com/news/home/20250517147002/en/Capital-One-Completes-Acquisition-of-Discover)¹) but also aligns them closely with each other in terms of market behavior and economic exposure. Given that [82% of American adults own a credit card](https://www.bankrate.com/credit-cards/news/credit-card-ownership-usage-statistics),² it's no surprise that these networks hold a prominent position in the stock market.

   Their structural similarities and high correlation make Visa and Mastercard ideal candidates for pairs trading, [a market-neutral strategy rooted in statistical arbitrage](https://www.sciencedirect.com/science/article/abs/pii/S1062940820301856).³ By analyzing historical price trends, correlation, and cointegration, this approach seeks to exploit pricing inefficiencies between the two stocks. Utilizing traditional pairs trading methods along with an efficient solution heuristic, investors may uncover opportunities for low-risk, profitable returns based on the consistent relationship between these two financial institutions.

## Tools 🛠️
This project leverages the following Python libraries and tools:

- **[yfinance](https://pypi.org/project/yfinance/):** For fetching historical stock price data from Yahoo Finance.
- **[NumPy](https://numpy.org/):** Used for efficient numerical computations and array operations.
- **[Pandas](https://pandas.pydata.org/):** Essential for data manipulation, time series handling, and analysis.
- **[Seaborn](https://seaborn.pydata.org/):** Provides high-level interface for attractive and informative statistical graphics.
- **[Matplotlib](https://matplotlib.org/):** Core plotting library used for customizing and visualizing financial data.

## Conclusions 💡
   The pair Visa (V) versus Mastercard (MA) performed the greatest in all three correlation coefficient models, scoring a rho of 95.62%, the greatest out all credit network pairs tested. The OLS regression model and residuals suggest a nonsteady, staggeringly large variance in price over the year, seen by Visa’s standard deviation of 33.60 and Mastercard’s standard deviation of 42.18 respectively. Thus, the pair was passed into the Dicky-Fuller test and produced a result constituting a t-statistic of -2.225 and p-value of 0.026. Given a negative t-statistic and p-value less than 0.05, the pair is likely stationary and the null hypothesis can be rejected with 95% confidence. Finally, after being passed through the optimized pairs trading model, a final cumulative return and profit margin of 28.0% was found. This statistic is carried by the associated Sharpe ratio of 0.648 and max drawdown of 2.4% respectively, indicating a decent yet non-optimal pairs trading model.

_Note: A Sharpe Ratio below one indicates a non-optimal risk-adjusted return, meaning risk may outweigh return. A max drawdown of less than 10% is often considered very good. A p-value for Dicky-Fuller < 0.05 rejects null hypothesis with 95% confidence, indicating moderate evidence of stationarity._

   Given high correlation between Visa and Mastercard, it can be confidently stated that the pair is correlated. This is reinforced by the fact that the pair is highly correlated throughout all correlative models utilized within this project. However, results of cointegration via Engle-Granger may be a more sufficient indicator of pairs trading success. In the cointegration model provided, the pair scored with a percentage of 2.6%, indicating that the pair is indeed stationary and fit for pairs trading, utilizing a non-constant and non-trending Dicky-Fuller model used for the high variance and standard deviation, as well as pattern seen in residual over OLS best fit line graph. Results received from risk evaluation statistics like Sharpe ratio (0.648) and max drawdown (0.024) indicate a decent, yet non-optimal model for pairs trading, which, with further improvement and optimization, may show greater returns and thus lower risk for investment. So, these findings support the initial hypothesis. It can now be concluded that Visa and Mastercard indeed exhibit high correlation, significant cointegration, and a profitable return with optimized pairs trading models. 

## Instructions 📝
### Method 1: Local reproduction
   1. Clone the repo
   ```bash 
      git clone https://github.com/jay-gleeson/fintech-pairs-csuf.git
      cd fintech-pairs-csuf
   ```
   2. Setup virtual environment.
   ```
      python -m venv venv
      venv\Scripts\activate
      pip install -r requirements.txt
   ```
   3. Run.
   ```
      python pairs_trading.py
   ```

### Method 2: Google Colab
1. Open .ipynb in Google Colab.
   See: [Open in Colab](archive/pairs_trading.ipynb).
2. Run Jupyter Notebook via Runtime >> Run All or Ctrl+F9.


## 📖 References
¹ Soheili, Sie, and Danielle Dietz. “Capital One Completes Acquisition of Discover.” _Business Wire_, 18 May 2025, [www.businesswire.com/news/home/20250517147002/en/Capital-One-Completes-Acquisition-of-Discover](https://www.businesswire.com/news/home/20250517147002/en/Capital-One-Completes-Acquisition-of-Discover).

² Martin, Erik J. "Credit card ownership and usage statistics." _Bankrate_. 21 December 2023, [www.bankrate.com/credit-cards/news/credit-card-ownership-usage-statistics/](https://www.bankrate.com/credit-cards/news/credit-card-ownership-usage-statistics).

³ Lin, Tsai-Yu, et al. “Multi-asset pair-trading strategy: A statistical learning approach.” _The North American Journal of Economics and Finance_, vol. 55, Jan. 2021, p. 101295, [doi.org/10.1016/j.najef.2020.101295](https://doi.org/10.1016/j.najef.2020.101295).
