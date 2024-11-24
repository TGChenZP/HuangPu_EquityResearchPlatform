# Project HuangPu: Equity Analysis Platform


## Requirements
Run ```requirements.txt```

## Instructions

1. Manually download interest rate data for Australia, US, China Mainland and China HKSAR from investing.com.

* You must manually select **Monthly** timeframe for 5 years, and do not alter the file name

* Note that the 10 year bond yield is an approximation to interest rates when calculating beta and alpha

   [Australia](https://au.investing.com/rates-bonds/australia-10-year-bond-yield-historical-data);
   [US](https://au.investing.com/rates-bonds/u.s.-10-year-bond-yield-historical-data); [Chinese Mainland](https://au.investing.com/rates-bonds/china-10-year-bond-yield-historical-data); [Hong Kong SAR](https://au.investing.com/rates-bonds/hong-kong-10-year-bond-yield-historical-data)


2. Run `stock_info.ipynb` to automatically retrieve asx company directories and asx gics

3. Run `analysis.ipynb`, changing TICKER, MARKET to match needs.

## Market Benchmarks

This document outlines the benchmarks used for comparisons across different markets. Each market is compared to the following global benchmarks:

- **^GSPC (S&P 500)**: A broad representation of the U.S. equity market.
- **GLD (Gold)**: A measure of gold prices, representing the precious metals market.
- **ACWI (MSCI ACWI Index)**: Tracks the performance of stocks from both developed and emerging markets worldwide.

Below are the benchmarks specific to each market, with highlights indicating the indices used for **Sharpe ratio calculations** in each region.



**1. United States (US)**
| Ticker       | Index Name                 | Description                                               |
|--------------|----------------------------|-----------------------------------------------------------|
| **^GSPC**    | S&P 500                    | _**Used for Sharpe ratio calculations in the U.S. market**_. Tracks 500 of the largest publicly traded companies in the U.S., representing various industries. It is a key benchmark for the overall U.S. stock market. |
| **^NDX**     | NASDAQ-100                 | Tracks the 100 largest non-financial companies on NASDAQ. |
| **^DJI**     | Dow Jones Industrial Average | Tracks 30 blue-chip U.S. companies.                      |
| **^W5000**   | Wilshire 5000 Total Market Index | Broad measure of the entire U.S. stock market.          |


**2. Australia (AU)**
| Ticker       | Index Name                 | Description                                               |
|--------------|----------------------------|-----------------------------------------------------------|
| **^AXJO**    | S&P/ASX 200                | Tracks the top 200 companies by market capitalization in Australia. |
| **^AXKO**    | S&P/ASX 300                | Extends the ASX 200 to include an additional 100 companies. |
| **^AORD**    | All Ordinaries             | _**Used for Sharpe ratio calculations in the Australian market**_. Tracks nearly all companies listed on the Australian Securities Exchange. |


**3. China (CN)**
| Ticker       | Index Name                 | Description                                               |
|--------------|----------------------------|-----------------------------------------------------------|
| **000300.SS**| CSI 300 Index              | _**Used for Sharpe ratio calculations in the Chinese market**_. Tracks the top 300 stocks from Shanghai and Shenzhen stock exchanges. |
| **399001.SZ**| Shenzhen Component Index   | Tracks 500 companies listed on the Shenzhen Stock Exchange. |
| **000001.SS**| SSE Composite Index        | Tracks all stocks listed on the Shanghai Stock Exchange.  |


**4. Hong Kong, China (HK)**
| Ticker       | Index Name                 | Description                                               |
|--------------|----------------------------|-----------------------------------------------------------|
| **^HSI**     | Hang Seng Index            | _**Used for Sharpe ratio calculations in the Hong Kong market**_. Tracks the largest 50 companies listed on the Hong Kong Stock Exchange. |
| **^HSCE**    | Hang Seng China Enterprises Index | Tracks mainland Chinese companies listed in Hong Kong. |
| **^HSCI**    | Hang Seng Composite Index  | Broad measure of the Hong Kong stock market.             |


**5. Global Indices (MSCI)**
| Ticker       | Index Name                 | Description                                               |
|--------------|----------------------------|-----------------------------------------------------------|
| **ACWI**     | MSCI ACWI Index            | _**Used for Sharpe ratio calculations in MSCI-based comparisons**_. Tracks the performance of stocks across 23 developed and 24 emerging markets worldwide, providing a comprehensive view of the global equity market. |
| **ACWX**     | MSCI ACWI ex USA Index     | Tracks the performance of stocks outside the U.S. market. |
| **EFA**      | MSCI EAFE Index            | Tracks large- and mid-cap stocks in developed markets outside the U.S. and Canada. |

**6. [GOLD]**

One must use either **US** or **MSCI** as market when analysing Gold ETFs

## Interest Rate
- MSCI class assets use US interest rates as a proxy

## Formulas used for metrics calculation

### 1. **Average Shareholder Equity**
   - **Calculation**: Average of the previous and current `Stockholder Equity`
   - **Explanation**: Average Shareholder Equity is used in calculating RoE, providing a more accurate view by averaging equity over time.

### 2. **Book Value**
   - **Formula**: `Total Assets - Total Liabilities Net Minority Interest`
   - **Explanation**: Book Value represents the value of a company if all assets were liquidated and liabilities paid. It’s used in P/B ratio calculations.

### 3. **Net Profit**
   - **Calculation**: Net Profit is calculated based on `Total Revenue` minus `Total Expenses` or other relevant costs.
   - **Explanation**: This metric is the basis for calculating margins and reflects the overall profitability.

### 4. **Net Profit Margin**
   - **Formula**: `(Net Profit / Total Revenue) * 100`
   - **Explanation**: Net Profit Margin measures the percentage of revenue that turns into profit after accounting for operating expenses. It’s useful for assessing profitability.

### 5. **Net Income Margin**
   - **Formula**: `(Net Income / Total Revenue) * 100`
   - **Explanation**: Net Income Margin shows the profit percentage after all expenses, taxes, and interest. A higher margin indicates better efficiency in generating profit.

### 6. **Return on Equity (RoE)**
   - **Formula**: `Net Profit / Average Shareholder Equity`
   - **Explanation**: RoE measures profitability relative to shareholder investments. It reflects a company’s efficiency in generating returns for its shareholders.

### 7. **Return on Assets (RoA)**
   - **Formula**: `Net Profit / Total Assets`
   - **Explanation**: RoA indicates how efficiently a company uses its assets to generate profit. Higher values suggest better asset utilization.

### 8. **Price-to-Earnings Ratio (P/E)**
   - **Formula**: `Last Close Price / (Net Income / Share Issued)`
   - **Explanation**: P/E ratio evaluates stock valuation by comparing the stock price to its earnings per share (EPS). A higher P/E may suggest overvaluation, while a lower P/E might indicate undervaluation.

### 9. **Price-to-Book Ratio (P/B)**
   - **Formula**: `Last Close Price / (Book Value / Share Issued)`
   - **Explanation**: P/B ratio compares a company's market value to its book value. It helps investors assess whether a stock is undervalued or overvalued relative to its assets.

### 10. **Dividends Per Share (DPS)**
   - **Explanation**: DPS represents the dividend a company pays per share. It is a direct indicator of income generated for shareholders.

### 11. **Dividend Yield**
   - **Formula**: `(DPS / Last Close Price) * 100`
   - **Explanation**: Dividend Yield shows the percentage return from dividends relative to the stock price. It's valuable for income-focused investors.

### 12. **Debt-to-Equity Ratio (D/E)**
   - **Formula**: `Total Debt / Stockholders Equity`
   - **Explanation**: D/E ratio indicates the relative proportion of debt and equity in financing the company's assets. A high ratio may signal potential risks with debt levels.

### 13. **Current Ratio**
   - **Formula**: `Current Assets / Current Liabilities`
   - **Explanation**: The Current Ratio measures a company’s ability to cover its short-term obligations with its short-term assets. A ratio above 1.0 is generally favorable.

### 14. **Interest Coverage Ratio**
   - **Formula**: `EBIT / Interest Expense`
   - **Explanation**: This ratio evaluates a company’s ability to cover interest expenses with its earnings before interest and taxes. A higher value suggests better debt management.

Metrics not found in this section are raw data from yfinance.

---

## Bibliography

- [Yahoo Finance](https://au.finance.yahoo.com/)