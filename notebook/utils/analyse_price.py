import pandas as pd
from utils.init import *


def get_prices(
    ticker: str,
    start_date: str,
    end_date: str = None,
    interval: str = "1d",
    price_df_list=None,
) -> pd.DataFrame:
    """
    Get historical prices for a given ticker.

    Parameters:
        - ticker (str): The ticker symbol of the stock.
        - start (str): The start date of the historical prices.
        - end_str (str): (default=None) The end date of the historical prices.
        - interval (str): (default='1d') The interval of the historical prices.
    """

    price_df_list = []

    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    start_year = int(start_date.split("-")[0])
    end_year = int(end_date.split("-")[0])

    # get prices for each year and concat to get the full price_df
    for year in range(start_year, end_year + 1):
        if year == start_year:
            price_df = yf.Ticker(ticker).history(
                interval=interval, start=start_date, end=f"{year}-12-31"
            )
        elif year == end_year:
            price_df = yf.Ticker(ticker).history(
                interval=interval, start=f"{year}-01-01", end=end_date
            )
        else:
            price_df = yf.Ticker(ticker).history(
                interval=interval, start=f"{year}-01-01", end=f"{year}-12-31"
            )

        price_df_list.append(price_df)

    price_df = pd.concat(price_df_list)

    try:
        price_df["Dividends"] = price_df["Dividends"].astype(float)
    except:
        pass

    return price_df


def get_return(price_df: pd.DataFrame, interest_rate: pd.DataFrame, interval: str = "M") -> pd.DataFrame:
    """
    Get return of stock on a specified interval basis (last close price/last close price previous period - 1)

    Parameters:
    - price_df: pd.DataFrame - DataFrame with stock prices (must have a 'Close' column and a DateTime index)
    - interest_rate: pd.DataFrame - DataFrame with interest rates
    - interval: str - Resampling interval ('M' for monthly, 'Q' for quarterly, 'Y' for yearly)

    Returns:
    - pd.DataFrame with stock returns and interest rate for each period
    """
    assert interval in ["M", "Q", "Y"], "interval must be M, Q or Y"

    # Ensure the index is a DatetimeIndex
    price_df.index = pd.to_datetime(price_df.index)

    # Resample to get the last close price of each period
    last_close_price = price_df.resample(interval).last()

    # Calculate the return
    return_series = last_close_price["Close"].pct_change() * 100
    return_series = return_series.to_frame(name=f"{interval}_Return (%)")

    # Format the index to display period correctly
    if interval == "Y":
        return_series.index = return_series.index.to_period("Y").strftime("%Y")
        return_series['merge_use_Y-M'] = return_series.index

        interest_rate['merge_use_Y-M'] = interest_rate.index.to_period(
            "Y").strftime("%Y")

    elif interval == "Q":
        return_series.index = return_series.index.to_period(
            "Q").strftime("%Y-Q%q")
        return_series['merge_use_Y-M'] = return_series.index

        interest_rate['merge_use_Y-M'] = interest_rate.index.to_period(
            "Q").strftime("%Y-Q%q")

    else:
        return_series.index = return_series.index.to_period(
            "M").strftime("%Y-%m")
        return_series['merge_use_Y-M'] = return_series.index

        interest_rate['merge_use_Y-M'] = interest_rate.index.to_period(
            "M").strftime("%Y-%m")

    return_series = return_series.merge(
        interest_rate, on='merge_use_Y-M', how="left").set_index(return_series.index)

    return_series.drop(columns='merge_use_Y-M', inplace=True)

    return_series[f"{interval}_Return - rf (%)"] = return_series[f"{interval}_Return (%)"] - \
        return_series['rf (%)']

    return return_series


def get_gics_industry_weighted_mean(
    return_df_dict: dict,
    TICKER: str,
    my_portfolio_tickers: list,
    same_gics_industry_weight_dict: dict,
    index_tickers: list,
    mode: str,
    comparable_tickers: list,
    **kwargs,
) -> pd.DataFrame:

    # Create dictionaries to store the weighted means for each component
    GICS_Industry_Weighted_Mean = {
        f"{mode}_Return - rf (%)": dd(float),
        f"{mode}_Return (%)": dd(float),
        "rf (%)": dd(float),
    }

    # Iterate through each date in the TICKER's return data (assumed main timeframe)
    for date in return_df_dict[TICKER].index:
        # Calculate the sum of weights excluding tickers with NaN for this date
        total_weight = 0
        adjusted_weight_dict = {}

        for ticker in my_portfolio_tickers:
            # Skip the index tickers and the main TICKER
            if ticker not in index_tickers + [TICKER]:
                # Check if the date exists in the ticker's DataFrame before accessing
                if date in return_df_dict[ticker].index:
                    # Check if the return value for this date is not NaN
                    if not pd.isna(return_df_dict[ticker].loc[date, f'{mode}_Return - rf (%)']):
                        # Sum the valid weights
                        total_weight += same_gics_industry_weight_dict[ticker]
                        adjusted_weight_dict[ticker] = same_gics_industry_weight_dict[ticker]

        # If total weight is not zero, normalize weights to sum to 100
        if total_weight > 0:
            for ticker in adjusted_weight_dict.keys():
                adjusted_weight_dict[ticker] /= total_weight

        # Calculate the weighted values for each required column
        for ticker, weight in adjusted_weight_dict.items():
            GICS_Industry_Weighted_Mean[f"{mode}_Return - rf (%)"][date] += (
                weight * return_df_dict[ticker].loc[date,
                                                    f'{mode}_Return - rf (%)']
            )
            GICS_Industry_Weighted_Mean[f"{mode}_Return (%)"][date] += (
                weight * return_df_dict[ticker].loc[date, f'{mode}_Return (%)']
            )
            GICS_Industry_Weighted_Mean['rf (%)'][date] += (
                weight * return_df_dict[ticker].loc[date, 'rf (%)']
            )

    # Create a DataFrame to store the results for each column
    weighted_mean_df = pd.DataFrame({f'{mode}_Return (%)': GICS_Industry_Weighted_Mean[f'{mode}_Return (%)'],
                                     f'{mode}_Return - rf (%)': GICS_Industry_Weighted_Mean[f'{mode}_Return - rf (%)'],
                                     'rf (%)': GICS_Industry_Weighted_Mean['rf (%)']})

    # Ensure every date from the TICKER's df is in the weighted mean df, adding NaN if not
    for date in return_df_dict[TICKER].index:
        if date not in weighted_mean_df.index:
            weighted_mean_df.loc[date] = np.nan

    # Sort index to ensure dates are in chronological order
    weighted_mean_df = weighted_mean_df.sort_index()

    # Assign the result back to the return_df_dict with appropriate key
    key = "GICS I.WMean" if "industry" in comparable_tickers["type"] else "GICS S.WMean"
    return_df_dict[key] = weighted_mean_df

    return return_df_dict


def get_monthly_stats(returns_df_dict: str, ticker: str, start_period: str, end_year: str, country: str):
    """ Get the mean, std, sharpe, beta, alpha of the stock """
    assert country in ['AU', 'US'], "Country must be AU or US"

    SHARPE_MONTHLY_MULTIPLIER = 12

    # get df for the period of interest
    period_of_interest_return_df = returns_df_dict[ticker].loc[start_period:end_year]

    stats_dict = {}

    # mean, std, n
    stats_dict["mean (%)"] = np.round(
        period_of_interest_return_df['M_Return (%)'].mean(), 2)
    stats_dict["std (%)"] = np.round(
        period_of_interest_return_df['M_Return (%)'].std(), 2)

    # mean - rf, std - rf (without risk free rate)
    stats_dict['mean (-rf) (%)'] = np.round(
        period_of_interest_return_df['M_Return - rf (%)'].mean(), 2)
    stats_dict['std (-rf) (%)'] = np.round(
        period_of_interest_return_df['M_Return - rf (%)'].std(), 2)

    # sharpe
    stats_dict["n"] = period_of_interest_return_df[
        ~period_of_interest_return_df[f"M_Return (%)"].isna()
    ].shape[0]
    stats_dict["sharpe"] = np.round(
        np.sqrt(SHARPE_MONTHLY_MULTIPLIER) *
        stats_dict["mean (-rf) (%)"] / stats_dict["std (-rf) (%)"], 2
    )

    # early return
    if stats_dict["n"] == 0:
        return stats_dict

    def get_linreg(returns_df_dict: dict, ticker: str):

        # get first and last date
        regression_start_period = returns_df_dict[ticker].index[1]
        regression_end_period = returns_df_dict[ticker].index[-1]

        # fit regression to get beta and alpha
        X = returns_df_dict["^AORD" if country == 'AU' else '^GSPC' if country == 'US' else None].loc[regression_start_period:regression_end_period][
            f"M_Return - rf (%)"
        ]
        y = returns_df_dict[ticker].loc[regression_start_period:regression_end_period][
            f"M_Return - rf (%)"
        ]
        y.rename(f"{ticker}_M_Return - rf (%)", inplace=True)
        X_y = pd.concat([X, y], axis=1)

        X_y.dropna(inplace=True)
        X = X_y[[X_y.columns[0]]]
        y = X_y[X_y.columns[1]]

        X = sm.add_constant(X)

        linreg = sm.OLS(y, X).fit()

        return linreg

    linreg = get_linreg(returns_df_dict, ticker)

    stats_dict["CAPM beta"] = np.round(linreg.params[f"M_Return - rf (%)"], 2)
    stats_dict["CAPM alpha"] = np.round(linreg.params["const"], 2)

    return stats_dict


def historical_corr(monthly_returns_df_dict: dict, start_period: str, end_year: str):
    """
    Get the historical correlation between two stocks.

    Parameters:
        - historical_returns: dict
        - start_period: str
        - end_year: str
    """

    period_of_interest_return_df = pd.DataFrame()

    for ticker, returns_df in monthly_returns_df_dict.items():
        returns_df = returns_df[['M_Return (%)']].loc[start_period:end_year]
        returns_df.rename(
            columns={returns_df.columns[0]: ticker}, inplace=True)

        period_of_interest_return_df = pd.merge(
            period_of_interest_return_df,
            returns_df,
            left_index=True,
            right_index=True,
            how="outer",
        )

    # calculate correlation: corr() function from pandas
    correlation_df = period_of_interest_return_df.corr()

    return correlation_df, period_of_interest_return_df


def plot_returns(
    monthly_returns_df_dict: dict,
    quarterly_returns_df_dict: dict,
    yearly_returns_df_dict: dict,
    ticker: str,
    first_end_of_quarter: str,
    last_end_of_quarter: str,
    underlying_ticker=None,
    **kwargs,
):

    # Create a figure and 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 8))

    interested_monthly_returns_df = monthly_returns_df_dict[ticker][['M_Return (%)']][
        monthly_returns_df_dict[ticker].index > first_end_of_quarter
    ]
    interested_quarterly_returns_df = quarterly_returns_df_dict[ticker][['Q_Return (%)']][
        quarterly_returns_df_dict[ticker].index > first_end_of_quarter
    ]
    interested_yearly_returns_df = yearly_returns_df_dict[ticker][['Y_Return (%)']][
        yearly_returns_df_dict[ticker].index > first_end_of_quarter
    ]

    # Plot monthly returns as a bar chart
    axs[0].bar(
        interested_monthly_returns_df.index,
        interested_monthly_returns_df.values.flatten(),
        label="Monthly Returns (%)",
        color="red",
    )
    axs[0].axhline(y=0, color="gray", lw=1, linestyle=":", linewidth=0.5)
    axs[0].set_title("Monthly Returns (%)")
    axs[0].set_ylabel("Returns (%)")
    axs[0].set_xticks(np.arange(0, len(interested_monthly_returns_df), 3))
    axs[0].tick_params(axis="x", rotation=90)

    # Plot quarterly returns as a bar chart
    axs[1].bar(
        interested_quarterly_returns_df.index,
        interested_quarterly_returns_df.values.flatten(),
        label="Quarterly Returns (%)",
        color="orange",
    )
    axs[1].axhline(y=0, color="gray", lw=1, linestyle=":", linewidth=0.5)
    axs[1].set_title("Quarterly Returns (%)")
    axs[1].set_ylabel("Returns (%)")
    axs[1].tick_params(axis="x", rotation=90)  # Rotate x-axis tick labels

    # Add text above the bars for quarterly returns
    for i, value in enumerate(interested_quarterly_returns_df.values.flatten()):
        axs[1].text(
            i, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8, rotation=45
        )

    # Plot yearly returns as a bar chart
    axs[2].bar(
        interested_yearly_returns_df.index,
        interested_yearly_returns_df.values.flatten(),
        label="Yearly Returns (%)",
        color="yellow",
    )
    axs[2].axhline(y=0, color="gray", lw=1, linestyle=":", linewidth=0.5)
    axs[2].set_title("Yearly Returns (%)")
    axs[2].set_ylabel("Returns (%)")
    axs[2].tick_params(axis="x", rotation=90)  # Rotate x-axis tick labels

    # Add text above the bars for quarterly returns
    for i, value in enumerate(interested_yearly_returns_df.values.flatten()):
        axs[2].text(
            i, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8, rotation=45
        )

    # Set x-label only on the last subplot
    axs[2].set_xlabel("Date")

    fig.suptitle(
        f"{ticker} Returns from {first_end_of_quarter} to {last_end_of_quarter}",
        fontsize=16,
    )

    # Rotate x-axis labels 90 degrees on the last subplot
    plt.setp(axs[2].xaxis.get_majorticklabels(), rotation=90)

    # Tight layout for better spacing between plots
    plt.tight_layout()

    if underlying_ticker != None:
        plt.savefig(f"../outputs/{underlying_ticker}_WMean_returns.png")
    else:
        plt.savefig(f"../outputs/{ticker}_returns.png")

    # Show the plot
    plt.show()


def plot_correlation(correlation_df: pd.DataFrame, ticker: str):
    """
    Plot the correlation matrix.

    Parameters:
        - correlation_df: pd.DataFrame
        - ticker: str, the stock ticker symbol for the plot title
    """
    plt.figure(figsize=(10, 8))
    # Set vmin and vmax to -1 and 1, respectively, and use the custom colormap
    plt.matshow(correlation_df, fignum=1, vmin=-1, vmax=1, cmap="RdBu")
    plt.xticks(range(len(correlation_df.columns)),
               correlation_df.columns, rotation=80)
    plt.yticks(range(len(correlation_df.columns)), correlation_df.columns)
    plt.colorbar()  # Colour bar with the custom colormap
    plt.title(f"{ticker} Monthly Return Correlation Matrix")

    # Save the plot as a PNG image
    plt.savefig(f"../outputs/{ticker}_correlation_matrix.png")
    plt.show()


def fetch_ticker_price(ticker: str, index_tickers: list, country: str) -> tuple:
    """
    Helper function to fetch the price data for a given ticker.
    """
    ticker_with_suffix = f"{ticker}.AX" if (ticker not in index_tickers and country == 'AU') else ticker if (
        ticker not in index_tickers and country == 'US') else ticker
    return ticker, get_prices(ticker_with_suffix, "2019-06-01")


def get_historical_prices(
    my_portfolio_tickers_list: list, index_tickers: str, country: str,  historical_prices_dict: list = None,
) -> dict:
    """
    Fetch historical prices for all tickers in the portfolio concurrently.

    Parameters:
        - my_portfolio_tickers: List of tickers to fetch.
        - index_tickers: List of index tickers to check for non '.AX' suffix.

    Returns:
        - A dictionary of historical prices for each ticker.
    """

    historical_prices_dict = {} if historical_prices_dict is None else historical_prices_dict

    with mp.Pool(mp.cpu_count()) as pool:
        # Use starmap to fetch prices concurrently
        results = pool.starmap(
            fetch_ticker_price,
            [(ticker, index_tickers, country)
             for ticker in my_portfolio_tickers_list],
        )

    # Combine results into a dictionary
    new_historical_prices_dict = {
        ticker: price_df for ticker, price_df in results}
    historical_prices_dict.update(new_historical_prices_dict)

    return historical_prices_dict


def filter_returns(returns_df, start_date):
    """
    Helper function to filter returns DataFrame based on a start date.
    """
    return returns_df[returns_df.index > start_date]


def get_common_index(*dfs):
    """
    Helper function to find the common index across multiple DataFrames.
    """
    common_index = dfs[0].index
    for df in dfs[1:]:
        common_index = common_index.union(df.index)
    return common_index


def plot_returns_comparative(
    monthly_returns_df_dict: dict,
    quarterly_returns_df_dict: dict,
    yearly_returns_df_dict: dict,
    TICKER: str,
    first_end_of_quarter: str,
    last_end_of_quarter: str,
    comparable_tickers: dict,
    country: str,
    **kwargs,
):
    assert country in ['AU', 'US'], "Country must be AU or US"

    # Define a consistent color scheme for all returns
    ticker_color = "red"
    industry_color = "blue"
    aord_color = "green"

    # Define a bar width
    bar_width = 0.2

    # Create a figure and 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(20, 20))

    # Filter data for the selected range
    interested_monthly_returns_df = filter_returns(
        monthly_returns_df_dict[TICKER][['M_Return (%)']], first_end_of_quarter
    )
    interested_quarterly_returns_df = filter_returns(
        quarterly_returns_df_dict[TICKER][[
            'Q_Return (%)']], first_end_of_quarter
    )
    interested_yearly_returns_df = filter_returns(
        yearly_returns_df_dict[TICKER][['Y_Return (%)']], first_end_of_quarter
    )

    if comparable_tickers is not None:
        # Filter for the industry weighted mean and AORD
        industry_key = (
            "GICS I.WMean" if "industry" in comparable_tickers["type"] else "GICS S.WMean"
        )
        industry_monthly_returns = filter_returns(
            monthly_returns_df_dict[industry_key][[
                'M_Return (%)']], first_end_of_quarter
        )
        industry_quarterly_returns = filter_returns(
            quarterly_returns_df_dict[industry_key][[
                'Q_Return (%)']], first_end_of_quarter
        )
        industry_yearly_returns = filter_returns(
            yearly_returns_df_dict[industry_key][[
                'Y_Return (%)']], first_end_of_quarter
        )

    aord_monthly_returns = filter_returns(
        monthly_returns_df_dict["^AORD" if country == 'AU' else '^GSPC' if country == 'US' else None][[
            'M_Return (%)']], first_end_of_quarter
    )
    aord_quarterly_returns = filter_returns(
        quarterly_returns_df_dict["^AORD" if country == 'AU' else '^GSPC' if country == 'US' else None][[
            'Q_Return (%)']], first_end_of_quarter
    )
    aord_yearly_returns = filter_returns(
        yearly_returns_df_dict["^AORD" if country == 'AU' else '^GSPC' if country == 'US' else None][[
            'Y_Return (%)']], first_end_of_quarter
    )

    ### Reindexing to ensure proper date alignment ###
    # Monthly
    common_monthly_index = get_common_index(
        interested_monthly_returns_df, industry_monthly_returns, aord_monthly_returns
    )
    interested_monthly_returns_df = interested_monthly_returns_df.reindex(
        common_monthly_index
    )
    industry_monthly_returns = industry_monthly_returns.reindex(
        common_monthly_index)
    aord_monthly_returns = aord_monthly_returns.reindex(common_monthly_index)

    # Plot Monthly Returns
    x_labels_monthly = common_monthly_index
    x_monthly = np.arange(len(x_labels_monthly))

    # Corrected x-tick position by centering bars
    tick_positions = x_monthly

    plot_bar_returns(
        axs[0],
        x_monthly - bar_width,
        interested_monthly_returns_df,
        bar_width,
        f"{TICKER} Monthly Returns (%)",
        ticker_color,
        x_labels_monthly,
    )
    if not np.isnan(industry_monthly_returns).all().all():
        plot_bar_returns(
            axs[0],
            x_monthly,
            industry_monthly_returns,
            bar_width,
            "GICS Industry Weighted Mean (%)",
            industry_color,
            x_labels_monthly,
        )
    plot_bar_returns(
        axs[0],
        x_monthly + bar_width,
        aord_monthly_returns,
        bar_width,
        "^AORD (%)",
        aord_color,
        x_labels_monthly,
    )

    axs[0].set_title("Monthly Returns (%)")
    axs[0].set_ylabel("Returns (%)")

    # Set the tick positions and labels
    axs[0].set_xticks(tick_positions)
    axs[0].set_xticklabels(x_labels_monthly, rotation=90)

    # Quarterly
    common_quarterly_index = get_common_index(
        interested_quarterly_returns_df,
        industry_quarterly_returns,
        aord_quarterly_returns,
    )
    interested_quarterly_returns_df = interested_quarterly_returns_df.reindex(
        common_quarterly_index
    )
    industry_quarterly_returns = industry_quarterly_returns.reindex(
        common_quarterly_index
    )
    aord_quarterly_returns = aord_quarterly_returns.reindex(
        common_quarterly_index)

    x_labels_quarterly = common_quarterly_index
    x_quarterly = np.arange(len(x_labels_quarterly))

    # Corrected x-tick position for quarterly returns
    tick_positions_quarterly = x_quarterly

    plot_bar_returns(
        axs[1],
        x_quarterly - bar_width,
        interested_quarterly_returns_df,
        bar_width,
        f"{TICKER} Quarterly Returns (%)",
        ticker_color,
        x_labels_quarterly,
    )
    if not np.isnan(industry_quarterly_returns).all().all():
        plot_bar_returns(
            axs[1],
            x_quarterly,
            industry_quarterly_returns,
            bar_width,
            "GICS Industry Weighted Mean (%)",
            industry_color,
            x_labels_quarterly,
        )
    plot_bar_returns(
        axs[1],
        x_quarterly + bar_width,
        aord_quarterly_returns,
        bar_width,
        "^AORD (%)",
        aord_color,
        x_labels_quarterly,
    )

    axs[1].set_title("Quarterly Returns (%)")
    axs[1].set_ylabel("Returns (%)")

    # Set the tick positions and labels
    axs[1].set_xticks(tick_positions_quarterly)
    axs[1].set_xticklabels(x_labels_quarterly, rotation=90)

    # Yearly
    common_yearly_index = get_common_index(
        interested_yearly_returns_df, industry_yearly_returns, aord_yearly_returns
    )
    interested_yearly_returns_df = interested_yearly_returns_df.reindex(
        common_yearly_index
    )
    industry_yearly_returns = industry_yearly_returns.reindex(
        common_yearly_index)
    aord_yearly_returns = aord_yearly_returns.reindex(common_yearly_index)

    x_labels_yearly = common_yearly_index
    x_yearly = np.arange(len(x_labels_yearly))

    # Corrected x-tick position for yearly returns
    tick_positions_yearly = x_yearly

    plot_bar_returns(
        axs[2],
        x_yearly - bar_width,
        interested_yearly_returns_df,
        bar_width,
        f"{TICKER} Yearly Returns (%)",
        ticker_color,
        x_labels_yearly,
    )
    if not np.isnan(industry_yearly_returns).all().all():
        plot_bar_returns(
            axs[2],
            x_yearly,
            industry_yearly_returns,
            bar_width,
            "GICS Industry Weighted Mean (%)",
            industry_color,
            x_labels_yearly,
        )
    plot_bar_returns(
        axs[2],
        x_yearly + bar_width,
        aord_yearly_returns,
        bar_width,
        "^AORD (%)",
        aord_color,
        x_labels_yearly,
    )

    axs[2].set_title("Yearly Returns (%)")
    axs[2].set_ylabel("Returns (%)")
    axs[2].set_xlabel("Date")

    # Set the tick positions and labels
    axs[2].set_xticks(tick_positions_yearly)
    axs[2].set_xticklabels(x_labels_yearly, rotation=90)

    # Set a title for the entire figure
    fig.suptitle(
        f"{TICKER} Returns from {first_end_of_quarter} to {last_end_of_quarter}",
        fontsize=16,
    )

    # Adjust layout for better spacing between subplots and suptitle
    # Adjust the top to create more space for suptitle
    plt.subplots_adjust(top=0.5, bottom=0.08)

    # Adjust layout for better spacing between subplots
    # Use tight_layout with rect to account for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(f"../outputs/{TICKER}_comparative_returns.png")

    plt.plot()


def plot_bar_returns(ax, x, data, bar_width, label, color, x_labels):
    """
    Helper function to plot bar charts for returns data.
    """
    ax.bar(x, data.values.flatten(), width=bar_width, label=label, color=color)
    ax.axhline(y=0, color="gray", lw=1, linestyle=":", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.legend()


def plot_close_price_with_dollar_lines(TICKER: str, historical_prices: dict):
    """
    Plots the closing price for a given ticker with horizontal dotted lines at each dollar interval.

    Parameters:
    - TICKER (str): The ticker symbol for the stock.
    - historical_prices (dict): A dictionary where the key is the ticker symbol, and the value is a DataFrame containing the stock price data with a 'Close' column.
    """

    # Create a new figure
    plt.figure(figsize=(12, 6))

    # Set the plot title
    plt.title(f"{TICKER} Close Price")

    # Plot the closing prices
    plt.plot(historical_prices[TICKER]["Close"], label=f"{TICKER} Close Price")

    # Plot horizontal lines at each dollar interval

    min_price = historical_prices[TICKER]["Close"].min()
    max_price = historical_prices[TICKER]["Close"].max()
    diff = max_price - min_price

    # Set range based on price difference
    if diff < 1:
        range_value = 0.1
    elif diff < 10:
        range_value = 1
    elif diff < 100:
        range_value = 10
    elif diff < 1000:
        range_value = 100
    elif diff < 10000:
        range_value = 1000
    else:
        range_value = 10000

    # Round min and max prices to the nearest multiple of range_value
    min_price = (np.floor(min_price / range_value) * range_value)
    max_price = (np.ceil(max_price / range_value) * range_value) + range_value

    # Plot horizontal lines at each interval
    for i in np.arange(min_price, max_price, range_value):
        plt.axhline(y=i, color="gray", linestyle=":", linewidth=0.5)

    plt.savefig(f"../outputs/{TICKER}_close_price.png")

    # Show the plot
    plt.show()


def get_same_industry_tickers_mcap(same_industry_ASX_tickers_mcap_df: pd.DataFrame) -> pd.DataFrame:
    """ Get the market cap of the same industry tickers
    NOTE: AU ONLY
    """
    try:
        same_industry_tickers_mcap_df = same_industry_ASX_tickers_mcap_df.sort_values(
            by='Market Cap', ascending=False)
    except KeyError:  # catch empty DataFrame error (no comparable tickers)
        same_industry_tickers_mcap_df = same_industry_ASX_tickers_mcap_df

    return same_industry_tickers_mcap_df


def get_AU_ticker_mv_df(asx_companies_directory_df: pd.DataFrame, ticker_of_interest: str) -> pd.DataFrame:
    """ Get the market cap of the ticker of interest
    NOTE: AU ONLY
    """

    ticker_mv_df = asx_companies_directory_df[asx_companies_directory_df['ASX code'] == ticker_of_interest][[
        'ASX code', 'Market Cap']]
    ticker_mv_df['Market Cap'] = ticker_mv_df['Market Cap'].apply(
        lambda x: round(x/1e9, 2))
    ticker_mv_df.rename(
        columns={'Market Cap': 'Market Cap ($bn)'}, inplace=True)

    return ticker_mv_df


def get_AU_ticker_proportion_of_market(ticker_mv_df: pd.DataFrame, same_industry_tickers_mcap_df: pd.DataFrame) -> float:
    """ Get the proportion of the market cap of the ticker of interest to the market cap of the same industry
    NOTE: AU ONLY
    """
    try:
        ticker_proportion_of_market = ticker_mv_df['Market Cap ($bn)'].values[0] / \
            (same_industry_tickers_mcap_df['Market Cap'].sum() / 1e9)
    except IndexError:
        ticker_proportion_of_market = np.nan

    return ticker_proportion_of_market


def get_same_AU_gics_industry_weight_dict(same_industry_tickers_mcap_df: pd.DataFrame) -> dict:
    """ Get the weight of the same industry tickers 
    NOTE: AU ONLY
    """

    try:
        same_gics_industry_weight_dict = same_industry_tickers_mcap_df.set_index('ASX code')[
            'weight'].to_dict()
    except KeyError:
        same_gics_industry_weight_dict = {}

    return same_gics_industry_weight_dict


def get_analysis_needed_ticker_list(interested_ticker: str, index_tickers_list: list, comparable_ASX_tickers_dict: dict = None) -> list:
    """ Get the list of tickers needed for analysis """
    analysis_needed_ticker_list = [interested_ticker]
    analysis_needed_ticker_list.extend([ticker.split('.')[
        0] for ticker in comparable_ASX_tickers_dict['list'] if ticker.split('.')[0] != interested_ticker])
    analysis_needed_ticker_list.extend(index_tickers_list)

    return analysis_needed_ticker_list


def get_historical_prices_for_interested_list(my_portfolio_tickers_list: list, index_tickers_list: list, historical_prices_dict: dict, country: str) -> dict:
    """ Get the historical prices for the interested list of tickers """

    historical_prices_dict = get_historical_prices(
        my_portfolio_tickers_list, index_tickers_list, country, historical_prices_dict)

    return historical_prices_dict


def get_same_gics_stats_df(interested_ticker: str, stats_df: pd.DataFrame, comparable_ASX_tickers_dict: dict, index_tickers_list: list, same_industry_tickers_mcap_df: pd.DataFrame) -> pd.DataFrame:
    """ Get the stats for the tickers in the same GICS industry"""

    same_gics_stats_df = stats_df[~stats_df.index.isin(
        index_tickers_list + [interested_ticker, 'GICS I.WMean' if 'industry' in comparable_ASX_tickers_dict['type'] else 'GICS S.WMean'])]

    same_gics_stats_df['ASX code'] = same_gics_stats_df.index

    same_gics_stats_df = same_gics_stats_df.merge(
        same_industry_tickers_mcap_df, on='ASX code')

    return same_gics_stats_df


def get_stats_df(interested_ticker: str, stats_df: dict, comparable_ASX_tickers_dict: dict, index_tickers_list: list, same_industry_tickers_mcap_df: pd.DataFrame) -> pd.DataFrame:
    """ Get the stats for the tickers in the same GICS industry"""

    def get_weighted_mean_row(same_gics_stats_df: pd.DataFrame) -> pd.DataFrame:
        weighted_mean_dict = {}
        for col in same_gics_stats_df.columns:
            if col in ['ASX code', 'weight']:
                continue
            weighted_mean_dict[col] = np.average(
                same_gics_stats_df[col], weights=same_gics_stats_df['weight'])

        weighted_mean_df = pd.DataFrame(weighted_mean_dict, index=[
                                        'GICS I.WMean (Macro)' if 'industry' in comparable_ASX_tickers_dict['type'] else 'GICS S.WMean (Macro)'])

        return weighted_mean_df

    try:
        same_gics_stats_df = get_same_gics_stats_df(interested_ticker,
                                                    stats_df, comparable_ASX_tickers_dict, index_tickers_list, same_industry_tickers_mcap_df)
        weighted_mean_df = get_weighted_mean_row(same_gics_stats_df)
        weighted_mean_df = weighted_mean_df.drop(columns=['Market Cap ($bn)'])
    except KeyError:
        weighted_mean_df = pd.DataFrame()
    except TypeError:
        weighted_mean_df = pd.DataFrame()

    stats_df = pd.concat(
        [stats_df, weighted_mean_df])

    return stats_df


def get_historical_dividends(TICKER: str, historical_prices_dict: dict, country: str):
    """ Get historical dividends for a given ticker """

    assert country in ['AU', 'US'], "Country must be AU or US"

    # Convert date to AEST directly using tz_convert, since the index is already timezone-aware
    historical_dividends = historical_prices_dict[TICKER]

    # Reset index to move 'Date' from index to a column
    historical_dividends = historical_dividends.reset_index()

    # Ensure the 'Date' column is in datetime format
    historical_dividends['Date'] = pd.to_datetime(historical_dividends['Date'])

    # Convert the 'Date' column to AEST
    historical_dividends['Date'] = historical_dividends['Date'].dt.tz_convert(
        'Australia/Sydney' if country == 'AU' else 'US/Eastern' if country == 'US' else None)

    # Now make it timezone unaware but still a timestamp
    historical_dividends['Date'] = historical_dividends['Date'].dt.tz_localize(
        None)

    # Display the dividends greater than 0 with the adjusted 'Date' column
    historical_dividends = historical_dividends[historical_dividends['Dividends'] > 0][[
        'Date', 'Dividends']]

    return historical_dividends


def plot_dividends(TICKER: str, historical_dividends: pd.DataFrame, historical_prices_dict: dict, country: str):

    assert country in ['AU', 'US'], "Country must be AU or US"

    # Assuming historical_dividends is your DataFrame for a specific TICKER
    # Convert the date to AEST
    historical_dividends = historical_prices_dict[TICKER]

    # Reset index to move 'Date' from index to a column
    historical_dividends = historical_dividends.reset_index()
    historical_dividends['Date'] = pd.to_datetime(historical_dividends['Date'])
    historical_dividends['Date'] = historical_dividends['Date'].dt.tz_convert(
        'Australia/Sydney' if country == 'AU' else 'US/Eastern' if country == 'US' else None)
    historical_dividends['Date'] = historical_dividends['Date'].dt.tz_localize(
        None)

    # Filter the data to include only the rows where Dividends are greater than 0
    dividends_df = historical_dividends[historical_dividends['Dividends'] > 0][[
        'Date', 'Dividends']]

    # Calculate the percentage change in dividends
    dividends_df['Dividend Change (%)'] = dividends_df['Dividends'].pct_change(
    ) * 100

    # Plot the data
    # Create a plot with a defined size
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Date vs Dividends on the primary y-axis
    ax1.plot(dividends_df['Date'], dividends_df['Dividends'],
             marker='o', linestyle='-', color='b', label='Dividends')

    # Set the title and labels for the primary y-axis
    ax1.set_title(f'{TICKER} Dividends and Dividend Change Percent Over Time',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Dividends', fontsize=12, color='b')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add grid for better readability
    ax1.grid(True)

    # Create a secondary y-axis to plot the dividend change percentage
    ax2 = ax1.twinx()

    # Plot Date vs Dividend Change (%) on the secondary y-axis
    ax2.plot(dividends_df['Date'], dividends_df['Dividend Change (%)'],
             marker='x', linestyle='--', color='r', label='Dividend Change (%)')

    # Set the labels for the secondary y-axis
    ax2.set_ylabel('Dividend Change (%)', fontsize=12, color='r')

    # Add legends to distinguish between the two y-axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Annotate the percentage change values on the plot
    for i, (date, change) in dividends_df[['Date', 'Dividend Change (%)']].dropna().iterrows():
        ax2.annotate(f'{change:.1f}%', xy=(date, change), xytext=(5, 5),
                     textcoords='offset points', fontsize=10, color='r')

    # Save the plot as an image if needed
    plt.savefig(
        f'../outputs/{TICKER}_dividends_and_change_over_time.png', bbox_inches='tight', dpi=300)

    # Optionally display the plot
    plt.show()


def get_historical_splits(TICKER: str, historical_prices_dict: dict, country: str):
    """ Get historical splits for a given ticker """

    assert country in ['AU', 'US'], "Country must be AU or US"
    # Convert date to AEST directly using tz_convert, since the index is already timezone-aware
    historical_splits = historical_prices_dict[TICKER]

    # Reset index to move 'Date' from index to a column
    historical_splits = historical_splits.reset_index()

    # Ensure the 'Date' column is in datetime format
    historical_splits['Date'] = pd.to_datetime(historical_splits['Date'])

    # Convert the 'Date' column to AEST
    historical_splits['Date'] = historical_splits['Date'].dt.tz_convert(
        'Australia/Sydney' if country == 'AU' else 'US/Eastern' if country == 'US' else None)

    # Now make it timezone unaware but still a timestamp
    historical_splits['Date'] = historical_splits['Date'].dt.tz_localize(
        None)

    # Display the dividends greater than 0 with the adjusted 'Date' column
    historical_splits = historical_splits[historical_splits['Stock Splits'] > 0][[
        'Date', 'Stock Splits']]

    return historical_splits


def plot_splits_over_time(TICKER: str, historical_prices_dict: dict, country: str):
    """ Plot stock splits over time for a given ticker """
    assert country in ['AU', 'US'], "Country must be AU or US"
    # Assuming historical_splits is your DataFrame for a specific TICKER
    # Convert the date to AEST
    historical_splits = historical_prices_dict[TICKER]

    # Reset index to move 'Date' from index to a column
    historical_splits = historical_splits.reset_index()

    # Ensure the 'Date' column is in datetime format
    historical_splits['Date'] = pd.to_datetime(historical_splits['Date'])

    # Convert the 'Date' column to AEST
    historical_splits['Date'] = historical_splits['Date'].dt.tz_convert(
        'Australia/Sydney' if country == 'AU' else 'US/Eastern' if country == 'US' else None)

    # Now make it timezone-unaware but still a timestamp
    historical_splits['Date'] = historical_splits['Date'].dt.tz_localize(None)

    # Filter the data to include only the rows where Stock Splits are greater than 0
    splits_df = historical_splits[historical_splits['Stock Splits'] > 0][[
        'Date', 'Stock Splits']]

    # Plot the data
    # Create a plot with a defined size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Date vs Stock Splits
    ax.plot(splits_df['Date'], splits_df['Stock Splits'],
            marker='o', linestyle='-', color='b')

    # Set the title and labels
    ax.set_title(f'{TICKER} Stock Splits Over Time',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Stock Splits', fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add grid for better readability
    ax.grid(True)

    # Save the plot as an image if needed
    plt.savefig(f'../outputs/{TICKER}_stock_splits_over_time.png',
                bbox_inches='tight', dpi=300)

    # Optionally display the plot
    plt.show()


def plot_key_ticker_stats_table(stats_df: pd.DataFrame, TICKER: str, comparable_ASX_tickers_dict: dict, index_tickers_list: list):
    # Assuming key_ticker_stats is already created from your DataFrame

    if comparable_ASX_tickers_dict is None:
        key_ticker_stats = stats_df.loc[[TICKER]+index_tickers_list]
    else:
        key_ticker_stats = stats_df.loc[[TICKER]+index_tickers_list + [
            'GICS I.WMean' if 'industry' in comparable_ASX_tickers_dict['type'] else 'GICS S.WMean']]

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 2))  # Adjust figure size as needed
    ax.axis('tight')
    ax.axis('off')

    # Create a table in the plot
    table = ax.table(cellText=key_ticker_stats.values,
                     colLabels=key_ticker_stats.columns,
                     rowLabels=key_ticker_stats.index,
                     cellLoc='center', loc='center')

    # Bold the header row
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # This selects the header row
            cell.set_text_props(fontweight='bold')

    # Set the title with bold font
    plt.title(f"{TICKER} Key Ticker Stats", fontsize=14, fontweight='bold')

    # Save the plot as an image
    plt.savefig(
        f'../outputs/{TICKER}_key_ticker_stats_table.png', bbox_inches='tight', dpi=300)

    # Optionally display the plot
    plt.show()


def plot_gics_mcap_weights(TICKER: str, same_industry_tickers_mcap_df: pd.DataFrame):

    try:
        # Create the plot
        fig, ax = plt.subplots(figsize=(6, 2))  # Adjust figure size as needed
        ax.axis('tight')
        ax.axis('off')

        same_industry_tickers_mcap_df['Market Cap'] = same_industry_tickers_mcap_df['Market Cap'].apply(
            lambda x: round(x/1e9, 2))
        same_industry_tickers_mcap_df.rename(
            columns={'Market Cap': 'Market Cap ($bn)'}, inplace=True)
        same_industry_tickers_mcap_df['weight'] = same_industry_tickers_mcap_df['weight'].apply(
            lambda x: round(x, 2))
        # Create a table in the plot
        table = ax.table(cellText=same_industry_tickers_mcap_df.values,
                         colLabels=same_industry_tickers_mcap_df.columns,
                         cellLoc='center', loc='center')

        # Bold the header row
        for key, cell in table.get_celld().items():
            if key[0] == 0:  # This selects the header row
                cell.set_text_props(fontweight='bold')

        # Save the plot as an image
        plt.savefig(f'../outputs/{TICKER}_same_industry_tickers_mcap_table.png',
                    bbox_inches='tight', dpi=300)

        # Optionally display the plot
        plt.show()
    except KeyError:
        plt.close()


def get_monthly_stats_for_all_tickers(monthly_returns_df_dict: dict, comparable_ASX_tickers_dict: dict, index_tickers_list: list, same_industry_tickers_mcap_df: pd.DataFrame, TICKER: str, first_end_of_quarter: str, last_end_of_quarter: str, COUNTRY: str) -> pd.DataFrame:
    """ Get the monthly returns for the interested ticker """
    stats_dict = {}
    for ticker in monthly_returns_df_dict:
        stats_dict[ticker] = get_monthly_stats(
            monthly_returns_df_dict, ticker, first_end_of_quarter, last_end_of_quarter, COUNTRY)
    stats_df = pd.DataFrame(stats_dict).T
    stats_df = get_stats_df(TICKER, stats_df, comparable_ASX_tickers_dict,
                            index_tickers_list, same_industry_tickers_mcap_df)
    return stats_df
