from utils.init import *
from utils.params import *


def get_index_tickers_list(market: str) -> list:

    assert market in AVAILABLE_MARKETS, 'market not supported'

    if market == 'AU':
        return au_index_tickers_list
    elif market == 'US':
        return us_index_tickers_list
    elif market == 'CN':
        return cn_index_tickers_list
    elif market == 'HK':
        return hk_index_tickers_list
    elif market == 'MSCI':
        return msci_index_tickers_list


def get_interest_rate(market: str) -> dict:
    """ Get interest rate data for US and AU """

    assert market in AVAILABLE_MARKETS, 'market not supported'

    interest_rate_df_dict = {}

    market_full_name = {'AU': 'Australia',
                        'US': 'United States', 'CN': 'China', 'HK': 'Hong Kong', 'MSCI': 'United States'}

    # open the file
    interest_rate_df = pd.read_csv(
        f'../data/{market_full_name[market]} 10-Year Bond Yield Historical Data.csv')

    interest_rate_df_dict['raw'] = interest_rate_df

    # make the last price of prev month the interest rate for next month (fully observable to investor at t0)
    interest_rate_df['rf (%)'] = interest_rate_df['Price'].shift(-1)

    # drop NA and only keep certain columns
    interest_rate_df = interest_rate_df[['Date', 'rf (%)']]
    interest_rate_df = interest_rate_df.dropna()

    # convert date to datetime
    interest_rate_df['Date'] = pd.to_datetime(
        interest_rate_df['Date'], dayfirst=True)

    def get_monthly_interest_rate_df(interest_rate_df: pd.DataFrame) -> pd.DataFrame:
        """ Get monthly interest rate from daily interest rate """

        interest_rate_monthly_df = copy.deepcopy(interest_rate_df)
        interest_rate_monthly_df['rf (%)'] = interest_rate_monthly_df['rf (%)'].apply(
            lambda x: 100*(np.power(1+x/100, 1/12)-1))

        interest_rate_monthly_df = interest_rate_monthly_df.set_index('Date')

        return interest_rate_monthly_df

    interest_rate_df_dict['monthly'] = get_monthly_interest_rate_df(
        interest_rate_df)

    def get_quarterly_interest_rate_df(interest_rate_df: pd.DataFrame) -> pd.DataFrame:
        """ Get quarterly interest rate from daily interest rate """

        interest_rate_quarterly_df = copy.deepcopy(interest_rate_df)
        # select_month
        interest_rate_quarterly_df = interest_rate_quarterly_df[interest_rate_quarterly_df['Date'].dt.month.isin([
            1, 4, 7, 10])]
        # compound the interest rate
        interest_rate_quarterly_df['rf (%)'] = interest_rate_quarterly_df['rf (%)'].apply(
            lambda x: 100*(np.power(1+x/100, 3/12)-1))

        interest_rate_quarterly_df = interest_rate_quarterly_df.set_index(
            'Date')

        return interest_rate_quarterly_df

    interest_rate_df_dict['quarterly'] = get_quarterly_interest_rate_df(
        interest_rate_df)

    def get_annual_interest_rate_df(interest_rate_df: pd.DataFrame) -> pd.DataFrame:
        """ Get annual interest rate from daily interest rate """

        interest_rate_annualy_df = copy.deepcopy(interest_rate_df)
        interest_rate_annualy_df = interest_rate_annualy_df[interest_rate_annualy_df['Date'].dt.month.isin([
            1])]
        # compound the interest rate
        interest_rate_annualy_df = interest_rate_annualy_df.set_index('Date')

        interest_rate_annualy_df['rf (%)'] = interest_rate_annualy_df['rf (%)'].apply(
            lambda x: 100*(np.power(1+x/100, 1)-1))

        return interest_rate_annualy_df

    interest_rate_df_dict['annualy'] = get_annual_interest_rate_df(
        interest_rate_df)

    return interest_rate_df_dict


def get_asx_companies_directory() -> pd.DataFrame:
    """ Get the ASX companies directory 
    NOTE: AU ONLY
    """

    # read in data
    asx_companies_directory_df = pd.read_csv(
        '../data/asx_companies_directory.csv')

    # clean Market Cap column
    asx_companies_directory_df['Market Cap'] = asx_companies_directory_df['Market Cap'].apply(
        lambda x: float(x) if x.isnumeric() else float(x.lower()) if 'E+' in x else np.nan)

    return asx_companies_directory_df


def get_asx_gics():
    """ Get the ASX GICS data 
    NOTE: AU ONLY
    """

    asx_gics_df = pd.read_csv('../data/asx_gics.csv')

    # add a few known missing sectors
    updates = AU_GICS_FILL

    for ticker, data in updates.items():
        asx_gics_df.loc[asx_gics_df['Ticker'] ==
                        ticker, 'Sector'] = data['Sector']
        asx_gics_df.loc[asx_gics_df['Ticker'] ==
                        ticker, 'Industry'] = data['Industry']

    # Unify the industry and sector names
    asx_gics_df['Industry'] = asx_gics_df['Industry'].apply(lambda x: ''.join(
        str(x).split('-')).replace(' ', '').replace('—', ''))

    asx_gics_df['Sector'] = asx_gics_df['Sector'].apply(lambda x: ''.join(
        str(x).split('-')).replace(' ', '').replace('—', ''))

    return asx_gics_df


def get_top_ASX_companies_list(asx_companies_directory_df: pd.DataFrame) -> dict:
    """ Get the list of top companies in the ASX: top 500, and 1b 
    NOTE: AU ONLY
    """

    AU_top_list_dict = {}

    market_cap_threshold = 1000000000

    # all companies in the top 500
    AU_top_list_dict['top_500'] = asx_companies_directory_df.sort_values(
        'Market Cap', ascending=False).head(500)['ASX code'].values + '.AX'

    # get all companies with market cap above 1 billion
    AU_top_list_dict['above_1b'] = asx_companies_directory_df[asx_companies_directory_df['Market Cap']
                                                              >= market_cap_threshold]['ASX code'].values + '.AX'

    return AU_top_list_dict


def _find_similar_type_AU_tickers(asx_gics: pd.DataFrame, gics_type: str, gics_name: str, comparable_universe: list) -> list:
    """ Find the tickers of companies in the same GICS sector or industry, given a comparable universe, gics_type and gics_name
    NOTE: AU ONLY
    """

    same_gics_tickers = asx_gics[(asx_gics[gics_type] == gics_name) & (
        asx_gics['Ticker'].isin(comparable_universe))]['Ticker'].values

    return same_gics_tickers


def get_same_gics_ASX_tickers(asx_gics_df: pd.DataFrame, ASX_ticker_gics_dict: dict, AU_top_list_dict: dict) -> dict:
    """ Find the tickers of companies in the same GICS sector or industry, given a comparable universe
    NOTE: AU ONLY
    """

    same_gics_ASX_tickers_dict = {}

    same_gics_ASX_tickers_dict['same_sector_1bn'] = _find_similar_type_AU_tickers(asx_gics_df,
                                                                                  gics_type='Sector', gics_name=ASX_ticker_gics_dict['Sector'], comparable_universe=AU_top_list_dict['above_1b'])
    same_gics_ASX_tickers_dict['same_industry_1bn'] = _find_similar_type_AU_tickers(asx_gics_df,
                                                                                    gics_type='Industry', gics_name=ASX_ticker_gics_dict['Industry'], comparable_universe=AU_top_list_dict['above_1b'])
    same_gics_ASX_tickers_dict['same_sector_500'] = _find_similar_type_AU_tickers(asx_gics_df,
                                                                                  gics_type='Sector', gics_name=ASX_ticker_gics_dict['Sector'], comparable_universe=AU_top_list_dict['top_500'])
    same_gics_ASX_tickers_dict['same_industry_500'] = _find_similar_type_AU_tickers(asx_gics_df,
                                                                                    gics_type='Industry', gics_name=ASX_ticker_gics_dict['Industry'], comparable_universe=AU_top_list_dict['top_500'])

    return same_gics_ASX_tickers_dict


def get_ASX_ticker_gics(ticker_of_interest: str, asx_gics_df: pd.DataFrame) -> dict:
    """ Get the GICS sector and industry of the ticker of interest
    NOTE: AU ONLY
    """

    ASX_ticker_gics_dict = {}

    try:
        ASX_ticker_gics_dict['Sector'] = asx_gics_df[asx_gics_df['Ticker']
                                                     == f'{ticker_of_interest}.AX']['Sector'].values[0]
        ASX_ticker_gics_dict['Industry'] = asx_gics_df[asx_gics_df['Ticker']
                                                       == f'{ticker_of_interest}.AX']['Industry'].values[0]
    except IndexError:
        ASX_ticker_gics_dict['Sector'] = 'Unknown'
        ASX_ticker_gics_dict['Industry'] = 'Unknown'

    return ASX_ticker_gics_dict


# def get_comparable_ASX_tickers(ticker_of_interest: str, same_gics_ASX_tickers: dict):
#     """ Get the comparable ASX tickers.
#     NOTE: AU ONLY
#     """
#     comparable_ASX_tickers_dict = {'type': 'same_industry_500', 'list': same_gics_ASX_tickers['same_industry_500']} if \
#         (len(same_gics_ASX_tickers['same_industry_500']) > 1 or len(same_gics_ASX_tickers['same_industry_500']) == 1 and f'{ticker_of_interest}.AX' not in same_gics_ASX_tickers['same_industry_500']) \
#         else {'type': 'same_sector_1bn', 'list': same_gics_ASX_tickers['same_sector_1bn']}

#     # remove self from comparable_tickers
#     comparable_ASX_tickers_dict['list'] = [
#         x for x in comparable_ASX_tickers_dict['list'] if x != f'{ticker_of_interest}.AX']

#     return comparable_ASX_tickers_dict

def get_comparable_ASX_tickers(ticker_of_interest: str, same_gics_ASX_tickers: dict):
    """ Get the comparable ASX tickers.
    NOTE: AU ONLY
    """
    comparable_ASX_tickers_dict = {
        'type': 'same_sector_500', 'list': same_gics_ASX_tickers['same_sector_500']}

    # remove self from comparable_tickers
    comparable_ASX_tickers_dict['list'] = [
        x for x in comparable_ASX_tickers_dict['list'] if x != f'{ticker_of_interest}.AX']

    return comparable_ASX_tickers_dict


def get_same_gics_ASX_MCap_weights(interested_ticker: str, asx_companies_director_df: pd.DataFrame, comparable_ASX_tickers_dict: dict) -> pd.DataFrame:
    """ Get the market cap and weight of the comparable ASX tickers 
    NOTE: AU ONLY
    """

    # get market caps and weight
    same_industry_ASX_tickers_mcap_df = asx_companies_director_df[((asx_companies_director_df['ASX code']+'.AX').isin(
        comparable_ASX_tickers_dict['list'])) & (asx_companies_director_df['ASX code'] != interested_ticker)][['ASX code', 'Market Cap']]

    # check for is integer
    same_industry_ASX_tickers_mcap_df = same_industry_ASX_tickers_mcap_df[same_industry_ASX_tickers_mcap_df['Market Cap'].apply(
        lambda x: isinstance(x, float))]

    try:
        # get weight
        same_industry_ASX_tickers_mcap_df['weight'] = same_industry_ASX_tickers_mcap_df['Market Cap'] / \
            same_industry_ASX_tickers_mcap_df['Market Cap'].sum()
    except KeyError:  # handle no comparable tickers
        return pd.DataFrame()

    return same_industry_ASX_tickers_mcap_df


def nearest_three_month(input_date=None):
    """
    Returns the date of the nearest three-month period for a given date, taking off one month.

    Args:
    - input_date (datetime.date or str, optional): The input date. If not provided, defaults to today's date.
      If input_date is a string, it should be in 'YYYY-MM-DD' format.

    Returns:
    - str: The nearest three-month period in 'YYYY-MM' format.
    """
    if input_date is None:
        date = dt.today()
    elif isinstance(input_date, str):
        date = dt.strptime(input_date, "%Y-%m-%d")
    else:
        date = input_date

    # Subtract one month
    year = date.year
    month = date.month

    if month in [1, 2, 3]:
        year -= 1
        nearest_month = 12
    elif month in [4, 5, 6]:
        nearest_month = 3
    elif month in [7, 8, 9]:
        nearest_month = 6
    elif month in [10, 11, 12]:
        nearest_month = 9

    return f"{year}-{nearest_month:02d}"


def get_first_end_of_quarter(last_end_of_quarter: str):
    """ take in format 'YYYY-MM' and subtract 4 years from the first quarter """

    year = int(last_end_of_quarter.split('-')[0])
    month = int(last_end_of_quarter.split('-')[1])

    year -= 4

    return f"{year}-{month:02d}"
