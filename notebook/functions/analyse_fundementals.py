from functions.init import *


def get_balance_sheet_df(object: yf.Ticker, TICKER: str) -> pd.DataFrame:
    balance_sheet_df_list = list()

    balance_sheet_rows = [
        "Total Debt",  # total debt
        "Stockholders Equity",  # total shareholder equity.
        "Share Issued",  # n shares
        "Current Liabilities",  # total current liabilities,
        "Current Assets",  # total current assets
    ]

    for balance_sheet_row in balance_sheet_rows:
        try:
            balance_sheet_row = object.balance_sheet.loc[balance_sheet_row]
            balance_sheet_df_list.append(balance_sheet_row)
        except KeyError:
            print(f"Missing data for {balance_sheet_row} for ticker {TICKER}")

    get_balance_sheet_df = pd.concat(balance_sheet_df_list, axis=1)

    return get_balance_sheet_df


def get_cashflow_df(object: yf.Ticker, TICKER: str) -> pd.DataFrame:

    cashflow_df_list = list()

    cashflow_rows = [
        "Free Cash Flow",  # cashflow
        "Interest Paid Supplemental Data",
        "Income Tax Paid Supplemental Data",
    ]

    for cashflow_row in cashflow_rows:
        try:
            cashflow_row = object.cashflow.loc[cashflow_row]
            cashflow_df_list.append(cashflow_row)

        except KeyError as e:
            print(f"{cashflow_row} not found for ticker {TICKER}")

    cashflow_df = pd.concat(cashflow_df_list, axis=1)

    return cashflow_df


def get_financials_df(object: yf.Ticker, TICKER: str) -> pd.DataFrame:
    financial_df_list = list()

    financial_rows = [
        "EBITDA",
        "EBIT",
        "Gross Profit",  # profit
        "Operating Expense",
        "Net Income",  # net income
        "Total Revenue",  # revenue
        "Interest Expense",  # interest expense
    ]

    for financial_row in financial_rows:
        try:
            finacial_row = object.financials.loc[financial_row]
            financial_df_list.append(finacial_row)
        except KeyError:
            print(f"Missing data for {financial_row} for ticker {TICKER}")

    financials_df = pd.concat(financial_df_list, axis=1)

    return financials_df


def create_compound_key_features(
    stock_fundementals: pd.DataFrame, TICKER: str
) -> pd.DataFrame:
    try:
        stock_fundementals["Net Profit"] = (
            stock_fundementals["Gross Profit"]
            - stock_fundementals["Operating Expense"]
            - stock_fundementals["Interest Paid Supplemental Data"]
            - stock_fundementals["Income Tax Paid Supplemental Data"]
        )
    except KeyError:
        stock_fundementals["Net Profit"] = np.nan
        print(f"Missing data for Net Profit for ticker {TICKER}")

    # get the mean of every two rows
    def get_avg_shareholder_equity(series):
        out = [np.nan]
        series = series.copy()
        series.fillna(series.values[1], inplace=True)
        for i in range(1, len(series)):
            out.append((series[i] + series[i - 1]) / 2)
        return out

    try:
        stock_fundementals["Average Shareholder Equity"] = get_avg_shareholder_equity(
            stock_fundementals["Stockholders Equity"]
        )
    except KeyError:
        stock_fundementals["Average Shareholder Equity"] = np.nan
        print(f"Missing data for Average Shareholder Equity for ticker {TICKER}")

    try:
        stock_fundementals["Total Asset"] = (
            stock_fundementals["Current Liabilities"]
            + stock_fundementals["Stockholders Equity"]
        )
    except KeyError:
        stock_fundementals["Total Asset"] = np.nan
        print(f"Missing data for Total Asset for ticker {TICKER}")

    return stock_fundementals


def get_dividend_on_dates(
    all_dividends_series: pd.Series, stock_fundamentals_dates: pd.Series
) -> pd.Series:

    # Remove zone info by normalizing the datetime index (if zone info is present)
    all_dividends_series.index = all_dividends_series.index.tz_localize(None)

    # Ensure that stock_fundamentals_dates are in datetime format
    stock_fundamentals_dates = pd.to_datetime(stock_fundamentals_dates)

    date_index = [stock_fundamentals_dates[0]]
    dividends_in_period = [np.nan]

    for i in range(1, len(stock_fundamentals_dates)):
        # Filter dividends within the range of the previous and current fundamental date
        date_filtered_dividend = all_dividends_series[
            (all_dividends_series.index >= stock_fundamentals_dates[i - 1])
            & (all_dividends_series.index < stock_fundamentals_dates[i])
        ]
        date_index.append(stock_fundamentals_dates[i])
        dividends_in_period.append(date_filtered_dividend.sum())

    return pd.DataFrame(dividends_in_period, index=date_index, columns=["Dividends"])


def find_last_close_price(ticker_prices, stock_fundementals_dates):
    """Find the close price on the day of the fundementals, or backtrack to find closest earlier price"""

    close_price_dict = {}

    for date in stock_fundementals_dates:
        if date in ticker_prices.index:
            close_price = ticker_prices.loc[date]["Close"]
            close_price_dict[date] = close_price
        elif date < ticker_prices.index[0]:
            close_price_dict[date] = np.nan
        else:
            old_date = date
            while date not in ticker_prices.index:
                try:
                    date = date - timedelta(days=1)
                except:
                    break
            close_price = ticker_prices.loc[date]["Close"]
            close_price_dict[old_date] = close_price

    return pd.DataFrame(close_price_dict, index=["Last Close Price"]).T


# get key statistics
def get_key_statistics(stock_fundementals: pd.DataFrame, ticker: str) -> pd.DataFrame:
    stock_fundementals["Net Profit Margin"] = (
        stock_fundementals["Net Profit"] / stock_fundementals["Total Revenue"]
    )
    stock_fundementals["Net Income Margin"] = (
        stock_fundementals["Net Income"] / stock_fundementals["Total Revenue"]
    )

    stock_fundementals["RoE"] = (
        stock_fundementals["Net Profit"]
        / stock_fundementals["Average Shareholder Equity"]
    )
    stock_fundementals["RoA"] = (
        stock_fundementals["Net Profit"] / stock_fundementals["Total Asset"]
    )

    stock_fundementals["P/E"] = stock_fundementals["Last Close Price"] / (
        stock_fundementals["Net Income"] / stock_fundementals["Share Issued"]
    )
    stock_fundementals["P/B"] = stock_fundementals["Last Close Price"] / (
        stock_fundementals["Stockholders Equity"] / stock_fundementals["Share Issued"]
    )

    stock_fundementals["DPS"] = stock_fundementals["Dividends"]

    stock_fundementals["D/E"] = (
        stock_fundementals["Total Debt"] / stock_fundementals["Stockholders Equity"]
    )

    stock_fundementals["Current Ratio"] = (
        stock_fundementals["Current Assets"] / stock_fundementals["Current Liabilities"]
    )

    try:
        stock_fundementals["Interest Coverage Ratio"] = (
            stock_fundementals["EBIT"]
            / stock_fundementals["Interest Paid Supplemental Data"]
        )
    except KeyError:
        stock_fundementals["Interest Coverage Ratio"] = np.nan
        print(f"Missing data for Interest Coverage Ratio for ticker {ticker}")

    return stock_fundementals


def get_key_interested_stat_pct_change(
    key_interested_stats: pd.DataFrame,
) -> pd.DataFrame:
    key_interested_stats_pct_change = key_interested_stats.pct_change() * 100
    key_interested_stats_pct_change.rename(
        columns={
            "Net Profit Margin": "Net Profit Margin (%)",
            "Net Income Margin": "Net Income Margin (%)",
            "RoE": "RoE (%)",
            "RoA": "RoA (%)",
            "P/E": "P/E (%)",
            "P/B": "P/B (%)",
            "D/E": "D/E (%)",
            "Current Ratio": "Current Ratio (%)",
            "Interest Coverage Ratio": "Interest Coverage Ratio (%)",
            "DPS": "DPS (%)",
        },
        inplace=True,
    )
    return key_interested_stats_pct_change


def preprocess_dividends(
    object, first_end_of_quarter: str, stock_fundementals_dates
) -> pd.DataFrame:
    """Process dividends to match our dates"""
    all_dividends_series = object.dividends
    all_dividends_series = all_dividends_series[
        all_dividends_series.index >= first_end_of_quarter
    ]
    dividend_in_period_series = get_dividend_on_dates(
        all_dividends_series, stock_fundementals_dates
    )
    return dividend_in_period_series


def preprocess_last_close_price(
    historical_prices: pd.DataFrame, TICKER: str, stock_fundementals_dates
) -> pd.DataFrame:

    ticker_prices = deepcopy(historical_prices[TICKER])
    ticker_prices.index = pd.to_datetime(ticker_prices.index).date
    last_close_price_df = find_last_close_price(ticker_prices, stock_fundementals_dates)
    return last_close_price_df


def get_other_features(
    object: yf.Ticker,
    stock_fundementals: pd.DataFrame,
    first_end_of_quarter: str,
    historical_prices: pd.DataFrame,
    TICKER: str,
) -> pd.DataFrame:
    """Get other features, dividends and last close price"""

    # get other features
    stock_fundementals = create_compound_key_features(stock_fundementals, TICKER)

    stock_fundementals_dates = stock_fundementals.index

    # get dividends
    dividend_in_period_series = preprocess_dividends(
        object, first_end_of_quarter, stock_fundementals_dates
    )

    # get last close price
    last_close_price_df = preprocess_last_close_price(
        historical_prices, TICKER, stock_fundementals_dates
    )

    # merge
    stock_fundementals = stock_fundementals.merge(
        last_close_price_df, left_index=True, right_index=True
    )
    stock_fundementals = stock_fundementals.merge(
        dividend_in_period_series, left_index=True, right_index=True
    )

    stock_fundementals = get_key_statistics(stock_fundementals, TICKER)

    return stock_fundementals


def get_fundemental_dfs(first_end_of_quarter, historical_prices, TICKER):

    object = yf.Ticker(f"{TICKER}.AX")

    # get balance sheet, concat, sort and change index to date
    balance_sheet_df = get_balance_sheet_df(object, TICKER)
    cashflow_df = get_cashflow_df(object, TICKER)
    financials_df = get_financials_df(object, TICKER)

    stock_fundementals = pd.concat(
        [balance_sheet_df, cashflow_df, financials_df], axis=1
    )

    # sort the rows by index
    stock_fundementals = stock_fundementals.sort_index()
    # change index to datetime (date)
    stock_fundementals.index = pd.to_datetime(stock_fundementals.index).date

    stock_fundementals = get_other_features(
        object, stock_fundementals, first_end_of_quarter, historical_prices, TICKER
    )

    stock_fundementals.index = pd.to_datetime(stock_fundementals.index)

    raw_stats = stock_fundementals.loc[
        :,
        ~stock_fundementals.columns.isin(
            [
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
            ]
        ),
    ]
    key_interested_stats = stock_fundementals[
        [
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
        ]
    ]
    key_interested_stats_pct_change = get_key_interested_stat_pct_change(
        key_interested_stats
    )

    return (
        raw_stats,
        key_interested_stats.astype(float).round(2),
        key_interested_stats_pct_change.astype(float).round(2),
    )
