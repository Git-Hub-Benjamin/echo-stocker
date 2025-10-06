#!/usr/bin/env python3
"""
Stock Correlation Analyzer
Author: Benjamin Funk
Date: 2025-10-05
Purpose: Read tickers from a text file, fetch their weekly data from Yahoo Finance,
and calculate correlations, R², p-values, and time-lagged (straddled) relationships.
"""

import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import csv
from scipy import stats
import time
import os


# =========================
#  CONFIGURATION
# =========================
START_DATE = "2020-11-30"
END_DATE = "2024-11-30"
OUTPUT_CSV = "stock_correlations.csv"
TICKER_FILE = "tickers.txt"
VALID_TICKER_FILE = "valid_tickers.txt"
MAX_STRADDLE_WEEKS = 4


# =========================
#  DATA FETCH HELPERS
# =========================
def get_weekly_prices(ticker: str, start_date: str, end_date: str) -> list | None:
    """Fetch weekly closing prices for a given ticker from Yahoo Finance."""
    try:
        # rate limiting safety
        time.sleep(0.1)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval="1wk")

        if df.empty or len(df) < 26:
            return None

        prices = df["Close"].tolist()
        return prices

    except Exception as e:
        print(f"[Error] {ticker}: {e}")
        return None


# =========================
#  CORE PROCESSING
# =========================
def populate_ticker_data(tickers, start_date, end_date, max_weeks):
    """Download weekly prices and create straddled versions."""
    ticker_data = {}
    for ticker in tickers:
        series_list = []
        for k in range(max_weeks + 1):
            start_shift = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(weeks=k)
            end_shift = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(weeks=k)
            prices = get_weekly_prices(ticker, start_shift.strftime("%Y-%m-%d"), end_shift.strftime("%Y-%m-%d"))
            if prices:
                series_list.append(prices)
            else:
                series_list = None
                break
        ticker_data[ticker] = series_list
        print(f"Fetched {ticker}: {'Valid' if series_list else 'Invalid'}")
    return ticker_data


def find_valid_tickers(ticker_data):
    """Filter tickers that successfully fetched data."""
    valid = [t for t, v in ticker_data.items() if v is not None]
    with open(VALID_TICKER_FILE, "w") as f:
        for t in valid:
            f.write(t + "\n")
    print(f"✅ Wrote {len(valid)} valid tickers to {VALID_TICKER_FILE}")
    return valid


def calc_stats(a_prices, b_prices):
    """Calculate correlation, R², and p-value."""
    corr = np.corrcoef(a_prices, b_prices)[0, 1]
    r_squared = corr**2
    n = len(a_prices)
    t_stat = corr * np.sqrt((n - 2) / (1 - corr**2))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    return corr, r_squared, p_value


# =========================
#  MAIN LOGIC
# =========================
def process_pairs(valid_tickers, ticker_data, max_weeks, output_file):
    """Compute pairwise straddled correlations across all valid tickers."""
    top_corrs = []
    total = len(valid_tickers)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ticker A", "Ticker B", "Weeks Lag", "Correlation", "R²", "P-Value"])

    count = 0
    for i, t_a in enumerate(valid_tickers):
        for j, t_b in enumerate(valid_tickers):
            if i >= j:
                continue  # skip duplicates
            if ticker_data[t_a] is None or ticker_data[t_b] is None:
                continue

            for k in range(max_weeks + 1):
                a_prices = ticker_data[t_a][0]  # baseline
                b_prices = ticker_data[t_b][k]  # shifted

                if len(a_prices) == len(b_prices):
                    try:
                        corr, r_sq, p_val = calc_stats(a_prices, b_prices)
                        if np.isnan(corr):
                            continue

                        record = (t_a, t_b, k, corr, r_sq, p_val)
                        top_corrs.append(record)

                        with open(output_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(record)

                    except Exception as e:
                        print(f"⚠️ Error {t_a}-{t_b}: {e}")
                        continue

            count += 1
            if count % 25 == 0:
                print(f"...Processed {count}/{total} base tickers")

    print("✅ Finished processing all pairs.")
    summarize_results(top_corrs, output_file)


def summarize_results(results, output_csv):
    """Summarize top 20 correlations, R², and significance."""
    print("\n=== Summary ===")

    sort_by_corr = sorted(results, key=lambda x: abs(x[3]), reverse=True)[:20]
    sort_by_rsq = sorted(results, key=lambda x: x[4], reverse=True)[:20]
    sort_by_pval = sorted(results, key=lambda x: x[5])[:20]

    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Top 20 | Strongest Correlations"])
        writer.writerows(sort_by_corr)
        writer.writerow([])
        writer.writerow(["Top 20 | Highest R²"])
        writer.writerows(sort_by_rsq)
        writer.writerow([])
        writer.writerow(["Top 20 | Lowest p-Values (Most Significant)"])
        writer.writerows(sort_by_pval)

    print(f"📁 Results saved to {output_csv}")


# =========================
#  ENTRY POINT
# =========================
def main():
    if not os.path.exists(TICKER_FILE):
        print(f"❌ Missing {TICKER_FILE}. Please create it with one ticker per line.")
        return

    with open(TICKER_FILE, "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    print(f"Loaded {len(tickers)} tickers from {TICKER_FILE}")

    max_weeks = input(f"Enter number of straddle weeks (default {MAX_STRADDLE_WEEKS}): ") or MAX_STRADDLE_WEEKS
    try:
        max_weeks = int(max_weeks)
    except ValueError:
        print("Invalid input. Using default 4 weeks.")
        max_weeks = MAX_STRADDLE_WEEKS

    # Validate end date vs today
    if datetime.strptime(START_DATE, "%Y-%m-%d") + timedelta(weeks=max_weeks) > datetime.now() - timedelta(weeks=1):
        print("⚠️ End date too near current date. Please reduce straddle weeks.")
        return

    print("\nFetching ticker data...")
    ticker_data = populate_ticker_data(tickers, START_DATE, END_DATE, max_weeks)

    valid_tickers = find_valid_tickers(ticker_data)

    print(f"\nBeginning correlation analysis with {len(valid_tickers)} valid tickers...")
    process_pairs(valid_tickers, ticker_data, max_weeks, OUTPUT_CSV)


if __name__ == "__main__":
    main()