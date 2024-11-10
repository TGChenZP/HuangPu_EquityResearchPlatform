from utils.init import *
from utils.params import FUNDAMENTALS_RAW_COLUMNS, CASHFLOW_ROWS, BALANCE_SHEET_ROWS, FUNDAMENTAL_ROWS


def get_balance_sheet_df(object: yf.Ticker, TICKER: str) -> pd.DataFrame:
    balance_sheet_df_list = list()

    for balance_sheet_row in BALANCE_SHEET_ROWS:
        try:
            balance_sheet_row = object.balance_sheet.loc[balance_sheet_row]
            balance_sheet_df_list.append(balance_sheet_row)
        except KeyError as e:
            print(
                f"Missing column from balance sheet for ticker {TICKER}: {e}")

    get_balance_sheet_df = pd.concat(balance_sheet_df_list, axis=1)

    return get_balance_sheet_df


def get_cashflow_df(object: yf.Ticker, TICKER: str) -> pd.DataFrame:

    cashflow_df_list = list()

    for cashflow_row in CASHFLOW_ROWS:
        try:
            cashflow_row = object.cashflow.loc[cashflow_row]
            cashflow_df_list.append(cashflow_row)

        except KeyError as e:
            print(
                f"Missing column from cashflow table for ticker {TICKER}: {e}")

    cashflow_df = pd.concat(cashflow_df_list, axis=1)

    return cashflow_df


def get_fundamental_cols_dfs(object: yf.Ticker, TICKER: str) -> pd.DataFrame:
    fundamentals_df_list = list()

    for col in object.financials.index:
        if 'Expense' in col and col not in FUNDAMENTAL_ROWS:
            FUNDAMENTAL_ROWS.append(col)

    for fundamentals_row in FUNDAMENTAL_ROWS:
        try:
            finacial_row = object.financials.loc[fundamentals_row]
            fundamentals_df_list.append(finacial_row)
        except KeyError as e:
            print(
                f"Missing data for financials table for ticker {TICKER}: {e}")

    fundamentals_df = pd.concat(fundamentals_df_list, axis=1)

    return fundamentals_df


def create_compound_key_features(
    stock_fundamentals: pd.DataFrame, TICKER: str
) -> pd.DataFrame:
    try:
        if 'Total Expenses' in stock_fundamentals.columns:
            stock_fundamentals["Net Profit"] = (
                stock_fundamentals['Total Revenue'] -
                stock_fundamentals['Total Expenses'] -
                stock_fundamentals['Tax Provision']
            )
        else:
            stock_fundamentals["Net Profit"] = (
                stock_fundamentals['Total Revenue'] -
                stock_fundamentals['Interest Expense'] -
                stock_fundamentals['Tax Provision']
            )

            for col in stock_fundamentals.columns:
                if 'Expense' in col \
                    and col not in ['Total Expenses', 'Interest Expense'] \
                        and 'Income Expense' not in col:
                    stock_fundamentals['Net Profit'] -= stock_fundamentals[col]

    except KeyError as e:
        stock_fundamentals["Net Profit"] = np.nan
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
        stock_fundamentals["Average Shareholder Equity"] = get_avg_shareholder_equity(
            stock_fundamentals["Stockholders Equity"]
        )
    except KeyError as e:
        stock_fundamentals["Average Shareholder Equity"] = np.nan
        print(
            f"Missing data for Average Shareholder Equity for ticker {TICKER}: {e}")

    try:
        stock_fundamentals['Book Value'] = stock_fundamentals['Total Assets'] - \
            stock_fundamentals['Total Liabilities Net Minority Interest']
    except KeyError as e:
        stock_fundamentals['Book Value'] = np.nan
        print(f"Missing data for Book Value for ticker {TICKER}: {e}")

    return stock_fundamentals


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


def find_last_close_price(ticker_prices, stock_fundamentals_dates):
    """Find the close price on the day of the fundamentals, or backtrack to find closest earlier price"""

    close_price_dict = {}

    for date in stock_fundamentals_dates:
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
def get_key_statistics(stock_fundamentals: pd.DataFrame, ticker: str) -> pd.DataFrame:
    try:
        stock_fundamentals["Net Profit Margin"] = 100*(
            # Net Profit: same as Net Income but after operating expenses but before taxes and interest
            stock_fundamentals["Net Profit"] /
            # Total Revenue: total income generated by operations, sales of goods, services before expenses are deducted
            stock_fundamentals["Total Revenue"].replace(0, np.nan)
        )
    except KeyError as e:
        stock_fundamentals["Net Profit Margin"] = np.nan
        print(f"Missing data for Net Profit Margin for ticker {ticker}: {e}")

    try:
        stock_fundamentals["Net Income Margin"] = 100*(
            # Net Income: total profit after all expenses, taxes, and interest have been deducted from revenue
            stock_fundamentals["Net Income"] /
            stock_fundamentals["Total Revenue"].replace(0, np.nan)
        )
    except KeyError as e:
        stock_fundamentals["Net Income Margin"] = np.nan
        print(f"Missing data for Net Income Margin for ticker {ticker}: {e}")

    try:
        stock_fundamentals["RoE"] = (
            stock_fundamentals["Net Profit"]
            / stock_fundamentals["Average Shareholder Equity"].replace(0, np.nan)
        )
    except KeyError as e:
        stock_fundamentals["RoE"] = np.nan
        print(f"Missing data for RoE for ticker {ticker}: {e}")

    try:
        stock_fundamentals["RoA"] = (
            stock_fundamentals["Net Profit"] /
            stock_fundamentals["Total Assets"].replace(0, np.nan)
        )
    except KeyError as e:
        stock_fundamentals["RoA"] = np.nan
        print(f"Missing data for RoA for ticker {ticker}: {e}")

    try:
        stock_fundamentals["P/E"] = stock_fundamentals["Last Close Price"] / (
            (stock_fundamentals["Net Income"] /
             stock_fundamentals["Share Issued"].replace(0, np.nan)).replace(0, np.nan)
        )
    except KeyError as e:
        stock_fundamentals["P/E"] = np.nan
        print(f"Missing data for P/E for ticker {ticker}: {e}")

    try:
        stock_fundamentals["P/B"] = stock_fundamentals["Last Close Price"] / (
            stock_fundamentals["Book Value"] /
            stock_fundamentals["Share Issued"].replace(0, np.nan)
        )
    except KeyError as e:
        stock_fundamentals["P/B"] = np.nan
        print(f"Missing data for P/B for ticker {ticker}: {e}")

    try:
        stock_fundamentals["DPS"] = stock_fundamentals["Dividends"]
    except KeyError as e:
        stock_fundamentals["DPS"] = np.nan
        print(f"Missing data for DPS for ticker {ticker}: {e}")

    try:
        stock_fundamentals['Dividend Yield'] = 100*(
            stock_fundamentals["DPS"] /
            stock_fundamentals["Last Close Price"].replace(0, np.nan)
        )
    except KeyError as e:
        stock_fundamentals['Dividend Yield'] = np.nan
        print(f"Missing data for Dividend Yield for ticker {ticker}: {e}")

    try:
        stock_fundamentals["D/E"] = (
            stock_fundamentals["Total Debt"] /
            stock_fundamentals["Stockholders Equity"].replace(0, np.nan)
        )
    except KeyError as e:
        stock_fundamentals["D/E"] = np.nan
        print(f"Missing data for D/E for ticker {ticker}: {e}")

    try:
        stock_fundamentals["Current Ratio"] = (
            stock_fundamentals["Current Assets"] /
            stock_fundamentals["Current Liabilities"].replace(0, np.nan)
        )
    except KeyError as e:
        stock_fundamentals["Current Ratio"] = np.nan
        print(f"Missing data for Current Ratio for ticker {ticker}: {e}")

    try:
        stock_fundamentals["Interest Coverage Ratio"] = (
            stock_fundamentals["EBIT"]
            / stock_fundamentals["Interest Expense"].replace(0, np.nan)
        )
    except KeyError as e:
        stock_fundamentals["Interest Coverage Ratio"] = np.nan
        print(
            f"Missing data for Interest Coverage Ratio for ticker {ticker}: {e}")

    return stock_fundamentals


def get_key_interested_fundamentals_stat_diff(
    key_interested_fundamentals_stats: pd.DataFrame,
) -> pd.DataFrame:
    key_interested_fundamentals_stats_diff = key_interested_fundamentals_stats.diff()
    key_interested_fundamentals_stats_diff.rename(
        columns={
            "Net Profit Margin": "Net Profit Margin (Δ)",
            "Net Income Margin": "Net Income Margin (Δ)",
            "RoE": "RoE (Δ)",
            "RoA": "RoA (Δ)",
            "P/E": "P/E (Δ)",
            "P/B": "P/B (Δ)",
            "D/E": "D/E (Δ)",
            "Current Ratio": "Current Ratio (Δ)",
            "Interest Coverage Ratio": "Interest Coverage Ratio (Δ)",
            "DPS": "DPS (Δ)",
            "Dividend Yield": "Dividend Yield (Δ)",
            "Free Cash Flow": "Free Cash Flow (Δ)",
        },
        inplace=True,
    )
    return key_interested_fundamentals_stats_diff.astype(float).round(2)


def preprocess_dividends(
    object, first_end_of_quarter: str, stock_fundamentals_dates
) -> pd.DataFrame:
    """Process dividends to match our dates"""
    all_dividends_series = object.dividends
    all_dividends_series = all_dividends_series[
        all_dividends_series.index >= first_end_of_quarter
    ]
    dividend_in_period_series = get_dividend_on_dates(
        all_dividends_series, stock_fundamentals_dates
    )
    return dividend_in_period_series


def preprocess_last_close_price(
    historical_prices: pd.DataFrame, TICKER: str, stock_fundamentals_dates
) -> pd.DataFrame:

    ticker_prices = deepcopy(historical_prices[TICKER])
    ticker_prices.index = pd.to_datetime(ticker_prices.index).date
    last_close_price_df = find_last_close_price(
        ticker_prices, stock_fundamentals_dates)

    return last_close_price_df


def get_other_features(
    object: yf.Ticker,
    stock_fundamentals: pd.DataFrame,
    first_end_of_quarter: str,
    historical_prices: pd.DataFrame,
    TICKER: str,
) -> pd.DataFrame:
    """Get other features, dividends and last close price"""

    # get other features
    stock_fundamentals = create_compound_key_features(
        stock_fundamentals, TICKER)

    stock_fundamentals_dates = stock_fundamentals.index

    # get dividends
    dividend_in_period_series = preprocess_dividends(
        object, first_end_of_quarter, stock_fundamentals_dates
    )

    # get last close price
    last_close_price_df = preprocess_last_close_price(
        historical_prices, TICKER, stock_fundamentals_dates
    )

    # merge
    if "Last Close Price" not in stock_fundamentals.columns:
        stock_fundamentals = stock_fundamentals.merge(
            last_close_price_df, left_index=True, right_index=True
        )

    if "Dividends" not in stock_fundamentals.columns:
        stock_fundamentals = stock_fundamentals.merge(
            dividend_in_period_series, left_index=True, right_index=True
        )

    stock_fundamentals = get_key_statistics(stock_fundamentals, TICKER)

    return stock_fundamentals


def get_fundamentals_dfs(first_end_of_quarter, historical_prices, TICKER, COUNTRY):

    assert COUNTRY in ["AU", "US"], "COUNTRY must be either 'AU' or 'US'"

    object = yf.Ticker(f"{TICKER}.AX" if COUNTRY ==
                       "AU" else TICKER if COUNTRY == "US" else TICKER)

    # get balance sheet, concat, sort and change index to date
    balance_sheet_df = get_balance_sheet_df(object, TICKER)
    cashflow_df = get_cashflow_df(object, TICKER)
    fundamentals_df = get_fundamental_cols_dfs(object, TICKER)

    stock_fundamentals = pd.concat(
        [balance_sheet_df, cashflow_df, fundamentals_df], axis=1
    )

    stock_fundamentals = process_stock_fundamentals(
        stock_fundamentals, object, first_end_of_quarter, historical_prices, TICKER)

    raw_fundamentals_stats = get_raw_stats(stock_fundamentals)

    key_interested_fundamentals_stats = get_key_interested_fundamentals_stats(
        stock_fundamentals)

    key_interested_fundamentals_stats_fundamentals_diff = get_key_interested_fundamentals_stat_diff(
        key_interested_fundamentals_stats
    )

    return (
        raw_fundamentals_stats,
        key_interested_fundamentals_stats.astype(float).round(2),
        key_interested_fundamentals_stats_fundamentals_diff.astype(
            float).round(2),
        object
    )


def process_stock_fundamentals(stock_fundamentals: pd.DataFrame, object: str, first_end_of_quarter: pd.Timestamp, historical_prices: dict, TICKER: str) -> pd.DataFrame:
    """ Change Index, and get other features """
    stock_fundamentals = stock_fundamentals.sort_index()
    # change index to datetime (date)
    stock_fundamentals.index = pd.to_datetime(stock_fundamentals.index).date

    stock_fundamentals = get_other_features(
        object, stock_fundamentals, first_end_of_quarter, historical_prices, TICKER
    )

    stock_fundamentals.index = pd.to_datetime(stock_fundamentals.index)
    return stock_fundamentals


def get_raw_stats(stock_fundamentals: pd.DataFrame) -> pd.DataFrame:
    # raw_stats = stock_fundamentals.loc[
    #     :,
    #     ~stock_fundamentals.columns.isin(
    #         FUNDAMENTALS_RAW_COLUMNS
    #     ),
    # ]
    raw_stats = copy.deepcopy(stock_fundamentals)
    return raw_stats


def get_key_interested_fundamentals_stats(stock_fundamentals: pd.DataFrame) -> pd.DataFrame:

    key_interested_fundamentals_stats = pd.DataFrame()

    for col in FUNDAMENTALS_RAW_COLUMNS:
        try:
            key_interested_fundamentals_stats[col] = stock_fundamentals[col]
        except KeyError as e:
            print(f"{col} not found in stock_fundamentals: {e}")

    return key_interested_fundamentals_stats


def interpolate_fundamentals_stats(raw_stats_list: list, key: str) -> pd.DataFrame:

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
            dates_halfyear = new_index[new_index <= dt]
            number_halfyear = len(dates_halfyear)

        else:
            dates_halfyear = new_index[(new_index <= dt) & (
                new_index > raw_stats_list[key].index[i-1])]
            number_halfyear = len(dates_halfyear)

        # now each row's values divide by number_halfyear
        row_values = raw_stats_list[key].iloc[i].values / number_halfyear

        for date in dates_halfyear:
            new_rows_dict[date] = row_values

    out = pd.DataFrame(
        new_rows_dict, index=raw_stats_list[key].columns).T

    return out


def agg_interpolated_fundamentals_stats(interpolated_fundamentals_stats: pd.DataFrame, interested_dates: list) -> pd.DataFrame:

    new_rows_dict = dict()

    for i, dt in enumerate(interested_dates):
        if i == 0:
            interested_rows = interpolated_fundamentals_stats[
                interpolated_fundamentals_stats.index <= dt]
        else:
            interested_rows = interpolated_fundamentals_stats[(interpolated_fundamentals_stats.index <= dt) & (
                interpolated_fundamentals_stats.index > interested_dates[i-1])]

        interested_rows = interested_rows.sum()

        new_rows_dict[dt] = interested_rows

    interested_fundamentals_stats = pd.DataFrame(
        new_rows_dict, index=interpolated_fundamentals_stats.columns).T

    return interested_fundamentals_stats


def get_weighted_fundamentals(stats_df_dict: dict, same_gics_industry_weight_dict: dict) -> pd.DataFrame:
    """ Get the weighted fundamentals of the key interested stats """

    example_df = stats_df_dict[list(stats_df_dict.keys())[
        0]]

    output_dict = dict()

    for date in example_df.index:

        output_dict[date] = dd(float)

        for column in FUNDAMENTALS_RAW_COLUMNS:

            weight = 0

            for key in stats_df_dict:
                try:

                    value = stats_df_dict[key].loc[date, column]

                except KeyError as e:
                    print(f"Column {column} not found for ticker {key}: {e}")
                    continue

                if not pd.isna(value) and value not in [np.inf, -np.inf]:
                    output_dict[date][column] += value * \
                        same_gics_industry_weight_dict[key]
                    weight += same_gics_industry_weight_dict[key]

            if weight != 0:
                output_dict[date][column] /= weight

        weighted_GICS_key_interested_fundamentals_stats = pd.DataFrame(
            output_dict)
        weighted_GICS_key_interested_fundamentals_stats = weighted_GICS_key_interested_fundamentals_stats.T

        for col in FUNDAMENTALS_RAW_COLUMNS:
            if col not in weighted_GICS_key_interested_fundamentals_stats.columns:
                weighted_GICS_key_interested_fundamentals_stats[col] = np.nan

    return weighted_GICS_key_interested_fundamentals_stats[FUNDAMENTALS_RAW_COLUMNS]


def plot_raw_fundamentals_stats_table(interested_ticker_raw_fundamentals_stats: pd.DataFrame, TICKER: str):
    # Format numbers in the DataFrame to have commas for thousands
    formatted_stats = interested_ticker_raw_fundamentals_stats.applymap(
        lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x
    )

    # Create the plot for the table
    # Adjust the figure size as needed
    fig, ax = plt.subplots(figsize=(18, 16))
    ax.axis('tight')
    ax.axis('off')

    # Create a table in the plot
    table = ax.table(cellText=formatted_stats.T.values,
                     colLabels=formatted_stats.T.columns,
                     rowLabels=formatted_stats.T.index,
                     cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)  # Set the font size to a larger value
    table.scale(1.2, 1.5)  # Adjust scaling to increase cell height and width

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


def plot_key_fundamentals_multipliers_table(interested_ticker_key_interested_fundamentals_stats: pd.DataFrame, TICKER: str):
    # Create the plot for the table

    formatted_stats = interested_ticker_key_interested_fundamentals_stats.applymap(
        lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x
    )

    fig, ax = plt.subplots(figsize=(18, 8))  # Adjust the figure size as needed
    ax.axis('tight')
    ax.axis('off')

    # Create a table in the plot
    table = ax.table(cellText=formatted_stats.T.values,
                     colLabels=formatted_stats.T.columns,
                     rowLabels=formatted_stats.T.index,
                     cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)  # Set the font size to a larger value
    table.scale(1.2, 1.5)  # Adjust scaling to increase cell height and width

    # Bold the header row
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # This selects the header row
            cell.set_text_props(fontweight='bold')

    # Set the title with bold font
    plt.title(f"{TICKER} Key fundamentals Multipliers",
              fontsize=16, fontweight='bold')

    # Save the plot as an image
    plt.savefig(
        f'../outputs/{TICKER}_interested_ticker_key_interested_stats.png', bbox_inches='tight', dpi=300)

    # Optionally display the plot
    plt.show()


def plot_key_fundamentals_multipliers_diff_table(interested_ticker_key_interested_fundamentals_stats_diff: pd.DataFrame, TICKER: str):

    formatted_stats = interested_ticker_key_interested_fundamentals_stats_diff.applymap(
        lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x
    )

    # Create the plot for the table
    fig, ax = plt.subplots(figsize=(18, 8))  # Adjust the figure size as needed
    ax.axis('tight')
    ax.axis('off')

    # Create a table in the plot
    table = ax.table(cellText=formatted_stats.T.values,
                     colLabels=formatted_stats.T.columns,
                     rowLabels=formatted_stats.T.index,
                     cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)  # Set the font size to a larger value
    table.scale(1.2, 1.5)  # Adjust scaling to increase cell height and width

    # Bold the header row
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # This selects the header row
            cell.set_text_props(fontweight='bold')

    # Set the title with bold font
    plt.title(f"{TICKER} Key Stats Δ", fontsize=16, fontweight='bold')

    # Save the plot as an image
    plt.savefig(
        f'../outputs/{TICKER}_interested_ticker_key_interested_stats_diff.png', bbox_inches='tight', dpi=300)

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
                diff_primary = (
                    (primary_data[i] - primary_data[i - 1]))
                ax.text(primary_data.index[i], primary_data[i],
                        f'{diff_primary:.1f}', color='black', fontsize=8, ha='center')
        except KeyError as e:
            print(f"Column {column} not found for ticker {TICKER}: {e}")

        # Plot the weighted GICS data if available
        try:
            label_suffix = 'GICS I.WMean' if 'industry' in compared_type else 'GICS S.WMean'
            gics_data = weighted_GICS_key_interested_stats[column]
            ax.plot(weighted_GICS_key_interested_stats.index,
                    gics_data, label=f'{TICKER} {label_suffix}')

            for i in range(1, len(gics_data)):
                diff_gics = (
                    (gics_data[i] - gics_data[i - 1]))
                ax.text(gics_data.index[i], gics_data[i],
                        f'{diff_gics:.1f}', color='gray', fontsize=8, ha='center')
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


def get_raw_fundamentals_stats(comparable_ASX_tickers_dict: dict, first_end_of_quarter: str, historical_prices_dict: dict, COUNTRY: str):
    """ Get raw fundamentals stats for all tickers in comparable_ASX_tickers_dict """

    raw_fundamentals_stats_dict = dict()
    key_interested_stats_dict = dict()
    object_dict = dict()

    for ticker in comparable_ASX_tickers_dict['list']:
        ticker = ticker.split('.')[0]
        print('\n', ticker)
        raw_stats, key_interested_stats, key_interested_stats_diff, object = get_fundamentals_dfs(
            first_end_of_quarter, historical_prices_dict, ticker, COUNTRY)

        raw_fundamentals_stats_dict[ticker] = raw_stats
        key_interested_stats_dict[ticker] = key_interested_stats
        object_dict[ticker] = object

    return raw_fundamentals_stats_dict, key_interested_stats_dict, object_dict


def interpolate_fundamentals(raw_fundamentals_stats_dict: dict):
    """ Interpolate fundamentals stats for all tickers in raw_fundamentals_stats_dict to half yearly blocks """

    interpolated_fundamentals_stats_dict = {}
    for key in raw_fundamentals_stats_dict.keys():
        interpolated_fundamentals_stats = interpolate_fundamentals_stats(
            raw_fundamentals_stats_dict, key)
        interpolated_fundamentals_stats_dict[key] = interpolated_fundamentals_stats

    return interpolated_fundamentals_stats_dict


def get_interested_fundamentals_dates(interested_ticker_key_interested_fundamentals_stats_diff: pd.DataFrame):
    """ Get the dates that the ticker of interest has fundamentals for, to agg the interpolated fundamentals stats """

    interested_dates = [pd.to_datetime(
        dt.strftime('%Y-%m')) for dt in interested_ticker_key_interested_fundamentals_stats_diff.index]

    return interested_dates


def agg_interpolated_fundamentals_stats_for_comparable_tickers(interpolated_fundamentals_stats_dict: dict, interested_dates: list):
    """ Aggregate interpolated fundamentals stats for all tickers in interpolated_fundamentals_stats_dict """

    agg_interpolated_fundamentals_stats_df_dict = {}
    for key in interpolated_fundamentals_stats_dict:
        agg_interpolated_fundamentals_stats_df = agg_interpolated_fundamentals_stats(
            interpolated_fundamentals_stats_dict[key], interested_dates)
        agg_interpolated_fundamentals_stats_df_dict[key] = agg_interpolated_fundamentals_stats_df

    return agg_interpolated_fundamentals_stats_df_dict


def get_agg_interpolated_fundamentals_stats(raw_fundamentals_stats_dict: dict, interested_ticker_key_interested_fundamentals_stats_diff: pd.DataFrame):
    """ Aggregate interpolated fundamentals stats for all tickers in interpolated_fundamentals_stats_dict """

    interpolated_fundamentals_stats_dict = interpolate_fundamentals(
        raw_fundamentals_stats_dict)

    interested_dates = get_interested_fundamentals_dates(
        interested_ticker_key_interested_fundamentals_stats_diff)

    agg_interpolated_fundamentals_stats_df_dict = agg_interpolated_fundamentals_stats_for_comparable_tickers(
        interpolated_fundamentals_stats_dict, interested_dates)

    return agg_interpolated_fundamentals_stats_df_dict


def get_key_interested_fundamentals_stats_for_comparable(agg_interpolated_fundamentals_stats_df_dict: pd.DataFrame, object_dict: dict, first_end_of_quarter: str, historical_prices_dict: dict):
    """ Get key interested fundamentals stats from the stock_fundamentals """

    key_interested_fundamentals_stats_dict = {}
    key_interested_fundamentals_stats_diff_dict = {}

    for key in agg_interpolated_fundamentals_stats_df_dict:

        # change index and get other features
        stock_fundamentals = process_stock_fundamentals(
            agg_interpolated_fundamentals_stats_df_dict[key], object_dict[key], first_end_of_quarter, historical_prices_dict, key)

        # get key interested stats
        key_interested_stats = get_key_interested_fundamentals_stats(
            stock_fundamentals)

        # get key interested stats pct change
        key_interested_stats_diff = get_key_interested_fundamentals_stat_diff(
            key_interested_stats
        )

        key_interested_fundamentals_stats_dict[key] = key_interested_stats
        key_interested_fundamentals_stats_diff_dict[key] = key_interested_stats_diff

    return key_interested_fundamentals_stats_dict, key_interested_fundamentals_stats_diff_dict
