from functions.init import *


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


def get_return(price_df: pd.DataFrame, interval: str = "M") -> pd.DataFrame:
    """
    Get return of stock on a monthly basis (last close price/last close price previous period - 1)

        - pride_df: pd.DataFrame
        - interval: str (default='M')
    """
    assert interval in ["M", "Q", "Y"], "interval must be M, Q or Y"

    # find the last close price of each period
    last_close_price = price_df.resample(interval).last()

    # calculate the return
    return_series = last_close_price["Close"].pct_change() * 100

    # turn into a dataframe
    return_series = return_series.to_frame()

    return_series.columns = [f"{interval}_Return (%)"]
    # set index to month
    if interval == "Y":
        return_series.index = return_series.index.strftime("%Y")
    elif interval == "Q":
        # I want it to show quarter (i.e. 2024-Q1)
        return_series.index = return_series.index.to_period("Q").strftime("%Y-Q%q")
    else:
        return_series.index = return_series.index.strftime("%Y-%m")

    return return_series


def get_stats(returns_df_dict: str, ticker: str, start_period: str, end_year: str):

    period_of_interest_return_df = returns_df_dict[ticker].loc[start_period:end_year]

    stats_dict = {}

    # mean, std, n
    stats_dict["mean (%)"] = np.round(period_of_interest_return_df.mean().values[0], 2)
    stats_dict["std (%)"] = np.round(period_of_interest_return_df.std().values[0], 2)

    # sharpe
    mode = returns_df_dict[ticker].columns[0].split("_")[0]
    stats_dict["n"] = period_of_interest_return_df[
        ~period_of_interest_return_df[f"{mode}_Return (%)"].isna()
    ].shape[0]
    sharpe_multiplier = 4 if mode == "Q" else 12 if mode == "M" else 1
    stats_dict["sharpe"] = np.round(
        np.sqrt(sharpe_multiplier) * stats_dict["mean (%)"] / stats_dict["std (%)"], 2
    )

    # earliest and latest date for ticker
    regression_start_period = returns_df_dict[ticker].index[1]
    regression_end_period = returns_df_dict[ticker].index[-1]

    # beta over this period
    X = returns_df_dict["^AORD"].loc[regression_start_period:regression_end_period][
        f"{mode}_Return (%)"
    ]
    y = returns_df_dict[ticker].loc[regression_start_period:regression_end_period][
        f"{mode}_Return (%)"
    ]
    y.rename(f"{ticker}_{mode}_Return (%)", inplace=True)
    X_y = pd.concat([X, y], axis=1)

    X_y.dropna(inplace=True)
    X = X_y[[X_y.columns[0]]]
    y = X_y[X_y.columns[1]]

    X = sm.add_constant(X)

    linreg = sm.OLS(y, X).fit()
    stats_dict["CAPM beta"] = np.round(linreg.params[f"{mode}_Return (%)"], 2)
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
        returns_df = returns_df.loc[start_period:end_year]
        returns_df.rename(columns={returns_df.columns[0]: ticker}, inplace=True)

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

    interested_monthly_returns_df = monthly_returns_df_dict[ticker][
        monthly_returns_df_dict[ticker].index > first_end_of_quarter
    ]
    interested_quarterly_returns_df = quarterly_returns_df_dict[ticker][
        quarterly_returns_df_dict[ticker].index > first_end_of_quarter
    ]
    interested_yearly_returns_df = yearly_returns_df_dict[ticker][
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
    plt.xticks(range(len(correlation_df.columns)), correlation_df.columns, rotation=80)
    plt.yticks(range(len(correlation_df.columns)), correlation_df.columns)
    plt.colorbar()  # Colour bar with the custom colormap
    plt.title(f"{ticker} Monthly Return Correlation Matrix")

    # Save the plot as a PNG image
    plt.savefig(f"../outputs/{ticker}_correlation_matrix.png")
    plt.show()


def fetch_ticker_price(ticker: str, index_tickers: list):
    """
    Helper function to fetch the price data for a given ticker.
    """
    ticker_with_suffix = f"{ticker}.AX" if ticker not in index_tickers else ticker
    return ticker, get_prices(ticker_with_suffix, "2019-06-01")


def get_historical_prices(
    my_portfolio_tickers: list, index_tickers: str, historical_prices: list = None
) -> dict:
    """
    Fetch historical prices for all tickers in the portfolio concurrently.

    Parameters:
        - my_portfolio_tickers: List of tickers to fetch.
        - index_tickers: List of index tickers to check for non '.AX' suffix.

    Returns:
        - A dictionary of historical prices for each ticker.
    """

    historical_prices = {} if historical_prices is None else historical_prices

    with mp.Pool(mp.cpu_count()) as pool:
        # Use starmap to fetch prices concurrently
        results = pool.starmap(
            fetch_ticker_price,
            [(ticker, index_tickers) for ticker in my_portfolio_tickers],
        )

    # Combine results into a dictionary
    historical_prices = {ticker: price_df for ticker, price_df in results}

    return historical_prices


# def get_gics_industry_weighted_mean(return_df_dict: dict, TICKER: str, my_portfolio_tickers: list, same_gics_industry_weight_dict: dict, index_tickers: list, mode: str, **kwargs) -> pd.DataFrame:

#     GICS_Industry_Weighted_Mean = dd(float)

#     # Iterate through each ticker in the portfolio
#     for ticker in my_portfolio_tickers:
#         # Skip the index tickers and the main TICKER
#         if ticker not in index_tickers + [TICKER]:
#             # Iterate through the dates of the return data for the ticker
#             for date in return_df_dict[ticker].index:
#                 # Check if the return value is not NaN

#                 if not pd.isna(return_df_dict[ticker].loc[date].values[0]):
#                     # Sum the weighted returns for the GICS industry by date
#                     GICS_Industry_Weighted_Mean[date] += same_gics_industry_weight_dict[ticker] * \
#                         return_df_dict[ticker].loc[date].values[0]

#     # Create a DataFrame to store the results
#     weighted_mean_df = pd.DataFrame.from_dict(
#         GICS_Industry_Weighted_Mean, orient='index', columns=[f'{mode}_Return (%)'])
#     return_df_dict['GICS Industry Weighted Mean'] = weighted_mean_df

#     # make sure every time index of TICKER df is in the return_df_dict, if not, add the timestamp with NaN value to GICS Industry Weighted Mean
#     for date in return_df_dict[TICKER].index:
#         if date not in weighted_mean_df.index:
#             weighted_mean_df.loc[date] = np.nan

#     return_df_dict['GICS Industry Weighted Mean'] = weighted_mean_df.sort_index()

#     return return_df_dict


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
    GICS_Industry_Weighted_Mean = dd(float)

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
                    if not pd.isna(return_df_dict[ticker].loc[date].values[0]):
                        # Sum the valid weights
                        total_weight += same_gics_industry_weight_dict[ticker]
                        adjusted_weight_dict[ticker] = same_gics_industry_weight_dict[
                            ticker
                        ]

        # If total weight is not zero, normalize weights to sum to 100
        if total_weight > 0:
            for ticker in adjusted_weight_dict.keys():
                adjusted_weight_dict[ticker] /= total_weight

        # Now calculate the weighted return for this date
        for ticker, weight in adjusted_weight_dict.items():
            GICS_Industry_Weighted_Mean[date] += (
                weight * return_df_dict[ticker].loc[date].values[0]
            )

    # Create a DataFrame to store the results
    weighted_mean_df = pd.DataFrame.from_dict(
        GICS_Industry_Weighted_Mean, orient="index", columns=[f"{mode}_Return (%)"]
    )

    # Ensure every date from the TICKER's df is in the weighted mean df, adding NaN if not
    for date in return_df_dict[TICKER].index:
        if date not in weighted_mean_df.index:
            weighted_mean_df.loc[date] = np.nan

    return_df_dict[
        "GICS I.WMean" if "industry" in comparable_tickers["type"] else "GICS S.WMean"
    ] = weighted_mean_df.sort_index()

    return return_df_dict


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
    **kwargs,
):
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
        monthly_returns_df_dict[TICKER], first_end_of_quarter
    )
    interested_quarterly_returns_df = filter_returns(
        quarterly_returns_df_dict[TICKER], first_end_of_quarter
    )
    interested_yearly_returns_df = filter_returns(
        yearly_returns_df_dict[TICKER], first_end_of_quarter
    )

    # Filter for the industry weighted mean and AORD
    industry_key = (
        "GICS I.WMean" if "industry" in comparable_tickers["type"] else "GICS S.WMean"
    )
    industry_monthly_returns = filter_returns(
        monthly_returns_df_dict[industry_key], first_end_of_quarter
    )
    industry_quarterly_returns = filter_returns(
        quarterly_returns_df_dict[industry_key], first_end_of_quarter
    )
    industry_yearly_returns = filter_returns(
        yearly_returns_df_dict[industry_key], first_end_of_quarter
    )

    aord_monthly_returns = filter_returns(
        monthly_returns_df_dict["^AORD"], first_end_of_quarter
    )
    aord_quarterly_returns = filter_returns(
        quarterly_returns_df_dict["^AORD"], first_end_of_quarter
    )
    aord_yearly_returns = filter_returns(
        yearly_returns_df_dict["^AORD"], first_end_of_quarter
    )

    ### Reindexing to ensure proper date alignment ###
    # Monthly
    common_monthly_index = get_common_index(
        interested_monthly_returns_df, industry_monthly_returns, aord_monthly_returns
    )
    interested_monthly_returns_df = interested_monthly_returns_df.reindex(
        common_monthly_index
    )
    industry_monthly_returns = industry_monthly_returns.reindex(common_monthly_index)
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
    aord_quarterly_returns = aord_quarterly_returns.reindex(common_quarterly_index)

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
    industry_yearly_returns = industry_yearly_returns.reindex(common_yearly_index)
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

    # Set the plot title
    plt.title(f"{TICKER} Close Price")

    # Plot the closing prices
    plt.plot(historical_prices[TICKER]["Close"], label=f"{TICKER} Close Price")

    # Plot horizontal lines at each dollar interval
    min_price = int(np.floor(historical_prices[TICKER]["Close"].min())) - 1
    max_price = int(np.ceil(historical_prices[TICKER]["Close"].max())) + 1

    for i in range(min_price, max_price):
        plt.axhline(y=i, color="gray", linestyle=":", linewidth=0.5)

    plt.savefig(f"../outputs/{TICKER}_close_price.png")

    # Show the plot
    plt.show()


# Example usage:
# plot_close_price_with_dollar_lines('AAPL', historical_prices)
