from utils.init import *


def create_pdf(
    got_fundamentals,
    TICKER,
    MARKET,
    ASX_ticker_gics_dict={'Sector': 'Unknown', 'Industry': 'Unknown'},
    comparable_ASX_tickers_dict=None,
    ticker_mv_df=None,
    same_industry_tickers_mcap_df=None,
    market_value_rank=None,
    **kwargs,
):

    pdf = FPDF()

    # TITLE
    # Add a page
    pdf.add_page()
    # Set font for the title
    pdf.set_font("Arial", "B", 16)  # Bold and size 16 for the title
    title = f"Stock Analysis of Ticker: {TICKER}"
    pdf.cell(200, 10, txt=title, ln=True, align="C")

    if MARKET == "AU" and not (
        ASX_ticker_gics_dict["Sector"] == "Unknown"
        and ASX_ticker_gics_dict["Industry"] == "Unknown"
    ):
        # Comparable Ticker Universe
        # Set font for the content
        pdf.set_font("Arial", "B", size=12)
        pdf.cell(200, 10, txt=f"Comparable tickers universe", ln=True, align="L")
        pdf.set_font("Arial", size=10)
        # text = f"{'same INDUSTRY of TOP500' if 'industry' in comparable_ASX_tickers_dict['type'] else 'same SECTOR of MCAP$1BN+'}"
        text = f"Sector: {ASX_ticker_gics_dict['Sector']}, Industry: {ASX_ticker_gics_dict['Industry']}"
        pdf.cell(200, 10, txt=text, ln=True, align="L")
        text = f"Market Value Rank in Market: {market_value_rank}"
        pdf.cell(200, 10, txt=text, ln=True, align="L")

        # INDUSTRY TICKER MCAP TABLE
        # Optional: Add a title before the image
        pdf.set_font("Arial", "B", 12)
        pdf.cell(
            200,
            10,
            txt=f"{TICKER} Same {'Industry' if 'industry' in comparable_ASX_tickers_dict['type'] else 'Sector'} Ticker MCAP Table",
            ln=True,
            align="L",
        )
        pdf.ln(5)  # Line break before adding the image
        pdf.image(
            f"../outputs/{TICKER}_same_industry_tickers_mcap_table.png",
            x=10,
            y=None,
            w=50,
        )

        # Retrieve and convert the Market Cap of the stock in question (from ticker_mv) to billions
        ticker_market_cap_billion = round(
            ticker_mv_df.iloc[0]["Market Cap ($bn)"], 2)
        ticker_proportion_of_market = (
            ticker_mv_df["Market Cap ($bn)"].values[0]
            / same_industry_tickers_mcap_df["Market Cap ($bn)"].sum()
        )
        universe_market_value_billion = round(
            same_industry_tickers_mcap_df["Market Cap ($bn)"].sum(), 2
        )
        pdf.set_font("Arial", size=10)
        pdf.cell(
            200,
            10,
            txt=f"{TICKER} has a market value of ${ticker_market_cap_billion}B and is {round(ticker_proportion_of_market, 2)} times the universe market value of ${universe_market_value_billion}B,",
            align="L",
            ln=True
        )

    # KEY STATS
    pdf.set_font("Arial", "B", 12)  # Optional: Add a title before the image
    pdf.cell(200, 10, txt=f"{TICKER} Key Stats", ln=True, align="L")
    pdf.ln(5)  # Line break before adding the image
    pdf.image(
        f"../outputs/{TICKER}_key_ticker_stats_table.png", x=10, y=None, w=175)

    # COMPARATIVE RETURNS PLOTS
    pdf.set_font("Arial", "B", 12)  # Optional: Add a title before the image
    pdf.cell(
        200, 10, txt=f"{TICKER} Comparative Returns Plot", ln=True, align="L")
    pdf.ln(5)  # Line break before adding the image
    pdf.image(
        f"../outputs/{TICKER}_comparative_returns.png", x=10, y=None, w=175)

    # TICKER RETURN CHART
    pdf.add_page()  # Add a new page for the plot
    pdf.set_font("Arial", "B", 12)  # Optional: Add a title before the image
    pdf.cell(200, 10, txt=f"{TICKER} Returns Chart", ln=True, align="L")
    pdf.ln(5)  # Line break before adding the image
    pdf.image(f"../outputs/{TICKER}_returns.png", x=10, y=None, w=100)

    if MARKET == "AU" and not (
        ASX_ticker_gics_dict["Sector"] == "Unknown"
        and ASX_ticker_gics_dict["Industry"] == "Unknown"
    ):
        # SECTOR/INDUSTRY WMEAN RETURNS CHART
        # Optional: Add a title before the image
        pdf.set_font("Arial", "B", 12)
        pdf.cell(
            200,
            10,
            txt=f"{TICKER} {'Industry' if 'industry' in comparable_ASX_tickers_dict['type'] else 'Sector'} Weighted Mean Returns Chart",
            ln=True,
            align="L",
        )
        pdf.ln(5)  # Line break before adding the image
        pdf.image(
            f"../outputs/{TICKER}_WMean_returns.png", x=10, y=None, w=100)

        # ^AORD CHART
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, txt=f"{TICKER} ^AORD Chart", ln=True, align="L")
        pdf.ln(5)  # Line break before adding the image
        pdf.image(f"../outputs/^AORD_returns.png", x=10, y=None, w=100)

    # CLOSE PRICE CHART
    pdf.add_page()  # Add a new page for the plot
    pdf.set_font("Arial", "B", 12)  # Optional: Add a title before the image
    pdf.cell(200, 10, txt=f"{TICKER} Close Price Chart", ln=True, align="L")
    pdf.ln(5)  # Line break before adding the image
    pdf.image(f"../outputs/{TICKER}_close_price.png", x=10, y=None, w=150)

    # DIVIDENDS
    pdf.set_font("Arial", "B", 12)  # Optional: Add a title before the image
    pdf.cell(200, 10, txt=f"{TICKER} Dividends", ln=True, align="L")
    pdf.ln(5)  # Line break before adding the image
    pdf.image(
        f"../outputs/{TICKER}_dividends_and_change_over_time.png", x=10, y=None, w=125
    )

    # key multipliers
    if got_fundamentals:
        pdf.add_page()

        # Optional: Add a title before the image
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, txt=f"{TICKER} Raw Stats", ln=True, align="L")
        pdf.image(
            f"../outputs/{TICKER}_interested_ticker_raw_stats.png", x=10, y=None, w=175
        )

        # Optional: Add a title before the image
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, txt=f"{TICKER} Key Multipliers", ln=True, align="L")
        pdf.ln(5)  # Line break before adding the image
        pdf.image(
            f"../outputs/{TICKER}_interested_ticker_key_interested_stats.png",
            x=10,
            y=None,
            w=175,
        )

        if MARKET == "AU":
            # gics multipliers
            pdf.image(
                f"../outputs/{TICKER} GICS {'I' if 'industry' in comparable_ASX_tickers_dict['type'] else 'S'}.WMean_interested_ticker_key_interested_stats.png",
                x=10,
                y=None,
                w=175,
            )

        # key multiplier pct change
        # Optional: Add a title before the image
        pdf.set_font("Arial", "B", 12)
        pdf.cell(
            200, 10, txt=f"{TICKER} Key Multipliers Pct Change", ln=True, align="L"
        )
        pdf.ln(5)
        pdf.image(
            f"../outputs/{TICKER}_interested_ticker_key_interested_stats_diff.png",
            x=10,
            y=None,
            w=175,
        )

        if MARKET == "AU":
            # gics multipliers pct change
            pdf.image(
                f"../outputs/{TICKER} GICS {'I' if 'industry' in comparable_ASX_tickers_dict['type'] else 'S'}.WMean_interested_ticker_key_interested_stats_diff.png",
                x=10,
                y=None,
                w=175,
            )

        # plots of key multipliers
        pdf.add_page()
        images = [
            f"../outputs/{TICKER}_P_B_comparison.png",
            f"../outputs/{TICKER}_P_E_comparison.png",
            f"../outputs/{TICKER}_Net Income Margin_comparison.png",
            f"../outputs/{TICKER}_Net Profit Margin_comparison.png",
            f"../outputs/{TICKER}_ROE_comparison.png",
            f"../outputs/{TICKER}_ROA_comparison.png",
            f"../outputs/{TICKER}_D_E_comparison.png",
            f"../outputs/{TICKER}_Current Ratio_comparison.png",
            f"../outputs/{TICKER}_Interest Coverage Ratio_comparison.png",
            f"../outputs/{TICKER}_DPS_comparison.png",
            f'../outputs/{TICKER}_Dividend Yield_comparison.png',
            f"../outputs/{TICKER}_Free Cash Flow_comparison.png",
        ]

        # Set image dimensions and spacing
        width = 60  # Adjust width to make images smaller
        height_spacing = 50  # Adjust vertical space between rows
        images_per_row = 3  # Number of images per row

        # Loop through images, placing them in a grid
        x, y = 10, 10  # Starting coordinates
        for i, image in enumerate(images):
            pdf.image(image, x=x, y=y, w=width)

            # Update x and y positions for next image
            if (i + 1) % images_per_row == 0:  # Move to the next row
                x = 10
                y += height_spacing
            else:
                x += width + 7.5  # Move to the next column

    # APPENDIX
    pdf.add_page()  # Add a new page for the appendix
    # Set font for the title
    pdf.set_font("Arial", "B", 12)  # Bold and size 16 for the title

    # APPENDIX TITLE
    title = f"Appendix"
    pdf.cell(200, 10, txt=title, ln=True, align="C")

    # CORRELATION
    pdf.set_font("Arial", "B", 12)
    pdf.cell(
        200, 10, txt=f"{TICKER} Correlation Matrix Plot", ln=True, align="L")
    pdf.ln(5)  # Line break before adding the image
    pdf.image(
        f"../outputs/{TICKER}_correlation_matrix.png", x=10, y=None, w=140)

    # SPLITS
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt=f"{TICKER} Splits", ln=True, align="L")
    pdf.ln(5)  # Line break before adding the image
    pdf.image(
        f"../outputs/{TICKER}_stock_splits_over_time.png", x=10, y=None, w=100)

    # Save the PDF after adding the image
    pdf.output(
        f"../reports/{TICKER}_{MARKET}_comparable_tickers_report_with_plot.pdf")
