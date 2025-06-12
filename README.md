# Visa (V) versus Mastercard (MA) Pairs Trading 💳
Pairs trading project to track stock data, correlation, cointegration, and perform pairs trading methods on Credit Networks like Visa and Mastercard. 🏦

Project originally completed by [Jay Gleeson](https://github.com/jay-gleeson/) and [Reyli Hernandez](https://github.com/rey-hern). This project was conducted with the help of the [CIC | PCUBED](https://www.fullerton.edu/ecs/cicpcubed/) organization at California State University, Fullerton.

## Background 🗣️
Visa (V) and Mastercard (MA) are two dominant players in the global payments industry, often exhibiting similar market behaviors due to their comparable business models and economic exposure. This project explores the relationship between their stock prices through pairs trading, a market-neutral strategy that identifies pricing inefficiencies between two correlated assets. By analyzing historical data, correlation, and cointegration, we aim to determine if a profitable trading strategy can be implemented using this financial technique.

## Tools 🛠️
This project leverages the following Python libraries and tools:

- **[yfinance](https://pypi.org/project/yfinance/):** For fetching historical stock price data from Yahoo Finance.
- **[NumPy](https://numpy.org/):** Used for efficient numerical computations and array operations.
- **[Pandas](https://pandas.pydata.org/):** Essential for data manipulation, time series handling, and analysis.
- **[Seaborn](https://seaborn.pydata.org/):** Provides high-level interface for attractive and informative statistical graphics.
- **[Matplotlib](https://matplotlib.org/):** Core plotting library used for customizing and visualizing financial data.

## Conclusions 💡
Our analysis confirms a strong historical correlation between Visa and Mastercard, making them viable candidates for pairs trading. 

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
1. Open .py in Google Colab.
   See: [Open in Colab](https://github.com/jay-gleeson/fintech-pairs-csuf/blob/main/archive/pairs_trading.ipynb).
2. Run Jupyter Notebook via Runtime >> Run All or Ctrl+F9.

## Presentation 🗨️
The presentation covers the motivation behind pairs trading, methodology used, data-driven findings, and practical implications for retail and institutional investors.

*** Will include presentation image once finalized. 

---

This project was completed for the 2025 CIC Transfer Pathways Summer Research Program at CSUF, a 7-week research experience.

## 📖 References
