from utils.init import *


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
        except KeyError as e:
            print(
                f"Missing data for {balance_sheet_row} for ticker {TICKER}: {e}")

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
            print(f"{cashflow_row} not found for ticker {TICKER}: {e}")

    cashflow_df = pd.concat(cashflow_df_list, axis=1)

    return cashflow_df


def get_fundemental_cols_dfs(object: yf.Ticker, TICKER: str) -> pd.DataFrame:
    fundementals_df_list = list()

    fundementals_rows = [
        "EBITDA",
        "EBIT",
        "Gross Profit",  # profit
        "Operating Expense",
        "Net Income",  # net income
        "Total Revenue",  # revenue
        "Interest Expense",  # interest expense
    ]

    for fundementals_row in fundementals_rows:
        try:
            finacial_row = object.financials.loc[fundementals_row]
            fundementals_df_list.append(finacial_row)
        except KeyError as e:
            print(
                f"Missing data for {fundementals_row} for ticker {TICKER}: {e}")

    fundementals_df = pd.concat(fundementals_df_list, axis=1)

    return fundementals_df


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
    except KeyError as e:
        stock_fundementals["Net Profit"] = np.nan
        print(f"Missing data for Net Profit for ticker {TICKER}: {e}")

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
    except KeyError as e:
        stock_fundementals["Average Shareholder Equity"] = np.nan
        print(
            f"Missing data for Average Shareholder Equity for ticker {TICKER}: {e}")

    try:
        stock_fundementals["Total Asset"] = (
            stock_fundementals["Current Liabilities"]
            + stock_fundementals["Stockholders Equity"]
        )
    except KeyError as e:
        stock_fundementals["Total Asset"] = np.nan
        print(
            f"Missing data for Total Asset for ticker {TICKER}: raw_stats_list")

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
    try:
        stock_fundementals["Net Profit Margin"] = (
            stock_fundementals["Net Profit"] /
            stock_fundementals["Total Revenue"].replace(0, np.nan)
        )
    except KeyError as e:
        stock_fundementals["Net Profit Margin"] = np.nan
        print(f"Missing data for Net Profit Margin for ticker {ticker}: {e}")

    try:
        stock_fundementals["Net Income Margin"] = (
            stock_fundementals["Net Income"] /
            stock_fundementals["Total Revenue"].replace(0, np.nan)
        )
    except KeyError as e:
        stock_fundementals["Net Income Margin"] = np.nan
        print(f"Missing data for Net Income Margin for ticker {ticker}: {e}")

    stock_fundementals["RoE"] = (
        stock_fundementals["Net Profit"]
        / stock_fundementals["Average Shareholder Equity"].replace(0, np.nan)
    )
    stock_fundementals["RoA"] = (
        stock_fundementals["Net Profit"] /
        stock_fundementals["Total Asset"].replace(0, np.nan)
    )

    stock_fundementals["P/E"] = stock_fundementals["Last Close Price"] / (
        stock_fundementals["Net Income"] /
        stock_fundementals["Share Issued"].replace(0, np.nan)
    )
    stock_fundementals["P/B"] = stock_fundementals["Last Close Price"] / (
        stock_fundementals["Stockholders Equity"] /
        stock_fundementals["Share Issued"].replace(0, np.nan)
    )

    stock_fundementals["DPS"] = stock_fundementals["Dividends"]

    try:
        stock_fundementals["D/E"] = (
            stock_fundementals["Total Debt"] /
            stock_fundementals["Stockholders Equity"].replace(0, np.nan)
        )
    except KeyError as e:
        stock_fundementals["D/E"] = np.nan
        print(f"Missing data for D/E for ticker {ticker}: {e}")

    stock_fundementals["Current Ratio"] = (
        stock_fundementals["Current Assets"] /
        stock_fundementals["Current Liabilities"].replace(0, np.nan)
    )

    try:
        stock_fundementals["Interest Coverage Ratio"] = (
            stock_fundementals["EBIT"]
            / stock_fundementals["Interest Paid Supplemental Data"].replace(0, np.nan)
        )
    except KeyError as e:
        stock_fundementals["Interest Coverage Ratio"] = np.nan
        print(
            f"Missing data for Interest Coverage Ratio for ticker {ticker}: {e}")

    return stock_fundementals


def get_key_interested_fundementals_stat_pct_change(
    key_interested_fundementals_stats: pd.DataFrame,
) -> pd.DataFrame:
    key_interested_fundementals_stats_pct_change = key_interested_fundementals_stats.pct_change() * \
        100
    key_interested_fundementals_stats_pct_change.rename(
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
    return key_interested_fundementals_stats_pct_change.astype(float).round(2)


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
    last_close_price_df = find_last_close_price(
        ticker_prices, stock_fundementals_dates)
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
    stock_fundementals = create_compound_key_features(
        stock_fundementals, TICKER)

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
    if "Last Close Price" not in stock_fundementals.columns:
        stock_fundementals = stock_fundementals.merge(
            last_close_price_df, left_index=True, right_index=True
        )

    if "Dividends" not in stock_fundementals.columns:
        stock_fundementals = stock_fundementals.merge(
            dividend_in_period_series, left_index=True, right_index=True
        )

    stock_fundementals = get_key_statistics(stock_fundementals, TICKER)

    return stock_fundementals


def get_fundementals_dfs(first_end_of_quarter, historical_prices, TICKER):

    object = yf.Ticker(f"{TICKER}.AX")

    # get balance sheet, concat, sort and change index to date
    balance_sheet_df = get_balance_sheet_df(object, TICKER)
    cashflow_df = get_cashflow_df(object, TICKER)
    fundementals_df = get_fundemental_cols_dfs(object, TICKER)

    stock_fundementals = pd.concat(
        [balance_sheet_df, cashflow_df, fundementals_df], axis=1
    )

    stock_fundementals = process_stock_fundementals(
        stock_fundementals, object, first_end_of_quarter, historical_prices, TICKER)

    raw_fundementals_stats = get_raw_stats(stock_fundementals)

    key_interested_fundementals_stats = get_key_interested_fundementals_stats(
        stock_fundementals)

    key_interested_fundementals_stats_fundementals_pct_change = get_key_interested_fundementals_stat_pct_change(
        key_interested_fundementals_stats
    )

    return (
        raw_fundementals_stats,
        key_interested_fundementals_stats.astype(float).round(2),
        key_interested_fundementals_stats_fundementals_pct_change.astype(
            float).round(2),
        object
    )


def process_stock_fundementals(stock_fundementals: pd.DataFrame, object: str, first_end_of_quarter: pd.Timestamp, historical_prices: dict, TICKER: str) -> pd.DataFrame:
    stock_fundementals = stock_fundementals.sort_index()
    # change index to datetime (date)
    stock_fundementals.index = pd.to_datetime(stock_fundementals.index).date

    stock_fundementals = get_other_features(
        object, stock_fundementals, first_end_of_quarter, historical_prices, TICKER
    )

    stock_fundementals.index = pd.to_datetime(stock_fundementals.index)
    return stock_fundementals


def get_raw_stats(stock_fundementals: pd.DataFrame) -> pd.DataFrame:
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
    return raw_stats


def get_key_interested_fundementals_stats(stock_fundementals: pd.DataFrame) -> pd.DataFrame:
    key_interested_fundementals_stats = stock_fundementals[
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
    return key_interested_fundementals_stats


def interpolate_fundementals_stats(raw_stats_list: list, key: str) -> pd.DataFrame:

    raw_stats_list[key].index = [pd.to_datetime(dt.strftime('%Y-%m'))
                                 for dt in raw_stats_list[key].index]

    new_index = pd.date_range(
        start=(raw_stats_list[key]).index[0] - pd.DateOffset(months=6), end=(raw_stats_list[key]).index[-1] + pd.DateOffset(months=6), freq='6M')

    new_index = np.array([pd.to_datetime(dt.strftime('%Y-%m'))
                          for dt in new_index])

    new_rows_dict = dict()

    for i, dt in enumerate(raw_stats_list[key].index):

        if i == 0:
            # find number of dates in the new index that are less than the current date
            dates_biannual = new_index[new_index <= dt]
            number_biannual = len(dates_biannual)

        else:
            dates_biannual = new_index[(new_index <= dt) & (
                new_index > raw_stats_list[key].index[i-1])]
            number_biannual = len(dates_biannual)

        # now each row's values divide by number_biannual
        row_values = raw_stats_list[key].iloc[i].values / number_biannual

        for date in dates_biannual:
            new_rows_dict[date] = row_values

    out = pd.DataFrame(
        new_rows_dict, index=raw_stats_list[key].columns).T

    return out


def agg_interpolated_fundementals_stats(interpolated_fundementals_stats: pd.DataFrame, interested_dates: list) -> pd.DataFrame:

    new_rows_dict = dict()

    for i, dt in enumerate(interested_dates):
        if i == 0:
            interested_rows = interpolated_fundementals_stats[
                interpolated_fundementals_stats.index <= dt]
        else:
            interested_rows = interpolated_fundementals_stats[(interpolated_fundementals_stats.index <= dt) & (
                interpolated_fundementals_stats.index > interested_dates[i-1])]

        interested_rows = interested_rows.sum()

        new_rows_dict[dt] = interested_rows

    interested_fundementals_stats = pd.DataFrame(
        new_rows_dict, index=interpolated_fundementals_stats.columns).T

    return interested_fundementals_stats


def get_weighted_fundementals(key_interested_fundementals_stats_pct_change_dict: dict, same_gics_industry_weight_dict: dict) -> pd.DataFrame:
    """ Get the weighted fundementals of the key interested stats """

    example_df = key_interested_fundementals_stats_pct_change_dict[list(key_interested_fundementals_stats_pct_change_dict.keys())[
        0]]

    output_dict = dict()

    for date in example_df.index:

        output_dict[date] = dd(float)

        for column in example_df.columns:

            weight = 0

            for key in key_interested_fundementals_stats_pct_change_dict:

                data = key_interested_fundementals_stats_pct_change_dict[key].loc[date, column]

                if not pd.isna(data) and data != np.inf:
                    output_dict[date][column] += data * \
                        same_gics_industry_weight_dict[key]
                    weight += same_gics_industry_weight_dict[key]

            if weight != 0:
                output_dict[date][column] /= weight
        weighted_GICS_key_interested_fundementals_stats_pct_change = pd.DataFrame(
            output_dict)
        weighted_GICS_key_interested_fundementals_stats_pct_change = weighted_GICS_key_interested_fundementals_stats_pct_change.T
    return weighted_GICS_key_interested_fundementals_stats_pct_change


def plot_raw_fundementals_stats_table(interested_ticker_raw_fundementals_stats: pd.DataFrame, TICKER: str):
    # Create the plot for the table
    fig, ax = plt.subplots(figsize=(14, 6))  # Adjust the figure size as needed
    ax.axis('tight')
    ax.axis('off')

    # Create a table in the plot
    table = ax.table(cellText=interested_ticker_raw_fundementals_stats.T.values,
                     colLabels=interested_ticker_raw_fundementals_stats.T.columns,
                     rowLabels=interested_ticker_raw_fundementals_stats.T.index,
                     cellLoc='center', loc='center')

    # Bold the header row
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # This selects the header row
            cell.set_text_props(fontweight='bold')

    # Set the title with bold font
    plt.title(f"{TICKER} Raw Stats", fontsize=16, fontweight='bold')

    # Save the plot as an image
    plt.savefig(
        f'../outputs/{TICKER}_interested_ticker_raw_stats.png', bbox_inches='tight', dpi=300)

    # Optionally display the plot
    plt.show()


def plot_key_fundementals_multipliers_table(interested_ticker_key_interested_fundementals_stats: pd.DataFrame, TICKER: str):
    # Create the plot for the table
    fig, ax = plt.subplots(figsize=(14, 3))  # Adjust the figure size as needed
    ax.axis('tight')
    ax.axis('off')

    # Create a table in the plot
    table = ax.table(cellText=interested_ticker_key_interested_fundementals_stats.T.values,
                     colLabels=interested_ticker_key_interested_fundementals_stats.T.columns,
                     rowLabels=interested_ticker_key_interested_fundementals_stats.T.index,
                     cellLoc='center', loc='center')

    # Bold the header row
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # This selects the header row
            cell.set_text_props(fontweight='bold')

    # Set the title with bold font
    plt.title(f"{TICKER} Key fundementals Multipliers",
              fontsize=16, fontweight='bold')

    # Save the plot as an image
    plt.savefig(
        f'../outputs/{TICKER}_interested_ticker_key_interested_stats.png', bbox_inches='tight', dpi=300)

    # Optionally display the plot
    plt.show()


def plot_key_fundementals_multipliers_pct_change_table(interested_ticker_key_interested_fundementals_stats_pct_change: pd.DataFrame, TICKER: str):
    # Create the plot for the table
    fig, ax = plt.subplots(figsize=(14, 3))  # Adjust the figure size as needed
    ax.axis('tight')
    ax.axis('off')

    # Create a table in the plot
    table = ax.table(cellText=interested_ticker_key_interested_fundementals_stats_pct_change.T.values,
                     colLabels=interested_ticker_key_interested_fundementals_stats_pct_change.T.columns,
                     rowLabels=interested_ticker_key_interested_fundementals_stats_pct_change.T.index,
                     cellLoc='center', loc='center')

    # Bold the header row
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # This selects the header row
            cell.set_text_props(fontweight='bold')

    # Set the title with bold font
    plt.title(f"{TICKER} Key Stats %Change", fontsize=16, fontweight='bold')

    # Save the plot as an image
    plt.savefig(
        f'../outputs/{TICKER}_interested_ticker_key_interested_stats_pct_change.png', bbox_inches='tight', dpi=300)

    # Optionally display the plot
    plt.show()


def plot_key_fundamentals_multipliers(interested_ticker_key_interested_fundamentals_stats: pd.DataFrame,
                                      weighted_GICS_key_interested_stats: pd.DataFrame,
                                      TICKER: str,
                                      compared_type: str):
    for column in interested_ticker_key_interested_fundamentals_stats.columns:
        fig, ax = plt.subplots(figsize=(6, 4))

        primary_data = None
        gics_data = None

        # Plot the primary ticker data if available
        try:
            primary_data = interested_ticker_key_interested_fundamentals_stats[column]
            ax.plot(interested_ticker_key_interested_fundamentals_stats.index,
                    primary_data, label=TICKER)

            for i in range(1, len(primary_data)):
                pct_change_primary = (
                    (primary_data[i] - primary_data[i - 1]) / primary_data[i - 1]) * 100
                ax.text(primary_data.index[i], primary_data[i],
                        f'{pct_change_primary:.1f}%', color='black', fontsize=8, ha='center')
        except KeyError as e:
            print(f"Column {column} not found for ticker {TICKER}: {e}")

        # Plot the weighted GICS data if available
        try:
            label_suffix = 'GICS I.WMean' if 'industry' in compared_type else 'GICS S.WMean'
            gics_data = weighted_GICS_key_interested_stats[column]
            ax.plot(weighted_GICS_key_interested_stats.index,
                    gics_data, label=f'{TICKER} {label_suffix}')

            for i in range(1, len(gics_data)):
                pct_change_gics = (
                    (gics_data[i] - gics_data[i - 1]) / gics_data[i - 1]) * 100
                ax.text(gics_data.index[i], gics_data[i],
                        f'{pct_change_gics:.1f}%', color='gray', fontsize=8, ha='center')
        except KeyError as e:
            print(f"Column {column} not found for GICS: {e}")

        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel(column)
        plt.title(f'{TICKER} {column} Comparison', fontsize=16)

        plt.xticks(rotation=45)
        plt.legend()

        # Replace any '/' in the column name to save the file properly
        safe_column_name = column.replace('/', '_')
        plt.savefig(
            f'../outputs/{TICKER}_{safe_column_name}_comparison.png', bbox_inches='tight', dpi=300)
        plt.show()
