# Proxy for 200, 300, small ord, all ords
AVAILABLE_MARKETS = ['AU', 'US', 'CN', 'HK', 'MSCI', 'GOLD']

AU_GICS_FILL = {
    'NWSLV.AX': {'Sector': 'Communication Services', 'Industry': 'Entertainment'},
    'KKC.AX': {'Sector': 'Financials', 'Industry': 'Diversified Financials'},
    'PCI.AX': {'Sector': 'Financials', 'Industry': 'Diversified Financials'},
    'RF1.AX': {'Sector': 'Financials', 'Industry': 'Investment Management'},
    'RG8.AX': {'Sector': 'Information Technology', 'Industry': 'Software'},
    'VG1.AX': {'Sector': 'Financials', 'Industry': 'Investment Management'},
    'WQG.AX': {'Sector': 'Financials', 'Industry': 'Investment Management'}
}

au_index_tickers_list = ['^AXJO', '^AXKO',
                         '^AORD', '^GSPC', 'ACWI', 'GLD']
us_index_tickers_list = ['^GSPC', '^NDX', '^DJI', '^W5000', 'ACWI', 'GLD']
cn_index_tickers_list = ['000300.SS', '399001.SZ',
                         '000001.SS', '^GSPC', 'ACWI', 'GLD']
hk_index_tickers_list = ['^HSI', '^HSCE', '^HSCI', '^GSPC', 'ACWI', 'GLD']
msci_index_tickers_list = ['ACWI', 'ACWX', 'EFA', '^GSPC', 'GLD']

FUNDAMENTALS_RAW_COLUMNS = [
    "Net Profit Margin",
    "Net Income Margin",
    "RoE",
    "RoA",
    "P/E",
    "P/B",
    "D/E",
    "Current Ratio",
    "Interest Coverage Ratio",
    "DPS",
    "Dividend Yield",
    "Free Cash Flow"
]

# from cashflow
CASHFLOW_ROWS = [
    "Free Cash Flow",  # cashflow
    "Interest Paid Supplemental Data",
    "Income Tax Paid Supplemental Data",
]

# from balance sheet
BALANCE_SHEET_ROWS = [
    "Total Debt",  # total debt
    "Stockholders Equity",  # total shareholder equity.
    "Share Issued",  # n shares
    "Current Liabilities",  # total current liabilities,
    "Current Assets",  # total current assets
    "Total Assets",  # total assets,
    "Total Liabilities Net Minority Interest",  # total liabilities
]

# from financials
FUNDAMENTAL_ROWS = [
    "EBITDA",
    "EBIT",
    "Gross Profit",  # profit
    "Operating Expense",
    "Net Income",  # net income
    "Total Revenue",  # revenue
    "Interest Expense",  # interest expense
    "Tax Provision",
    "Total Expenses"

]
