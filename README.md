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

3. Run `analysis.ipynb`, changing TICKER, MARKET and end of quarter data to match needs

## Metrics Calculation

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


---

Bibliography:

- [Yahoo Finance](https://au.finance.yahoo.com/)