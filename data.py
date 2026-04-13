"""
data.py — Data loading module.

Downloads historical stock data from Yahoo Finance using yfinance.
"""

import yfinance as yf
import pandas as pd


# Default tickers and date range
TICKERS = ["AAPL", "JPM", "TSLA"]
MARKET_TICKERS = ["^VIX", "^GSPC"]
DEFAULT_START = "2016-01-01"
DEFAULT_END = "2025-12-31"


def download_ticker(ticker: str, start: str = DEFAULT_START,
                    end: str = DEFAULT_END) -> pd.DataFrame:
    """Download daily OHLCV data for a single ticker.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [Open, High, Low, Close, Volume] and a
        DatetimeIndex.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                     progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    return df


def download_all(tickers: list[str] | None = None,
                 start: str = DEFAULT_START,
                 end: str = DEFAULT_END,
                 include_market: bool = True) -> dict[str, pd.DataFrame]:
    """Download data for multiple tickers.

    Parameters
    ----------
    tickers : list[str] | None
        Stock tickers. Defaults to TICKERS.
    start, end : str
        Date range.
    include_market : bool
        If True, also download VIX and S&P 500.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker symbol to its DataFrame.
    """
    if tickers is None:
        tickers = TICKERS

    all_tickers = list(tickers)
    if include_market:
        all_tickers += MARKET_TICKERS

    data: dict[str, pd.DataFrame] = {}
    for t in all_tickers:
        print(f"  Downloading {t} …")
        data[t] = download_ticker(t, start=start, end=end)
    return data


def load_data(tickers: list[str] | None = None,
              start: str = DEFAULT_START,
              end: str = DEFAULT_END) -> dict[str, pd.DataFrame]:
    """High-level helper: download data and forward-fill missing values.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    data = download_all(tickers, start=start, end=end)
    for t, df in data.items():
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        data[t] = df
    return data
