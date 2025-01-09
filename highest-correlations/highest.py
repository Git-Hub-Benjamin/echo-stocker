import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def get_weekly_prices(ticker: str, start_date: str, end_date: str) -> list:
    """
    Get weekly stock prices and returns for a given ticker and date range.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1wk')
        
        if df.empty or len(df) < 26:  # Minimum 26 weeks of data
            return None
        
        prices = df['Close'].tolist()
        return prices, df.index.tolist()  # Return both prices and dates
    
    except Exception as e:
        return None, None

pairs = [
    ('PHM', 'MHO', 'https://tinyurl.com/yhakxtv5'),
    ('PHM', 'TOL', 'https://tinyurl.com/4wpxuphh'),
    ('LEN', 'DHI', 'https://tinyurl.com/am4xmfkm'),
    ('CRON', 'TLRY', 'https://tinyurl.com/4ud2ae2j'),
    ('NVDA', 'APH', 'https://tinyurl.com/3jdzf9h5'),
    ('LEN', 'PHM', 'https://tinyurl.com/3ajuaseu'),
    ('DHI', 'MHO', 'https://tinyurl.com/h8fewktk'),
    ('GE', 'TOL', 'https://tinyurl.com/39enhbzf'),
    ('ACB', 'IDEX', 'https://tinyurl.com/z3b4mawk'),
    ('SNPS', 'CDNS', 'https://tinyurl.com/mv2uktac'),
    ('TMHC', 'PHM', 'https://tinyurl.com/yc8b3ffa'),
    ('ETN', 'PH', 'https://tinyurl.com/2p8wzxp3'),
    ('ETN', 'APH', 'https://tinyurl.com/y56wvetn'),
    ('KBH', 'TOL', 'https://tinyurl.com/jvcx5wfm'),
    ('CRS', 'WMT', 'https://tinyurl.com/nbhx8988'),
    ('PHM', 'DHI', 'https://tinyurl.com/ywyr4asn'),
    ('TOL', 'MHO', 'https://tinyurl.com/yc5uz5c7'),
    ('CRON', 'IDEX', 'https://tinyurl.com/umsyx8j4'),
    ('NVDA', 'AVGO', 'https://tinyurl.com/yypwtenb'),
    ('LEN', 'TOL', 'https://tinyurl.com/wcuk3avx')
]

class Colors:
    BLUE = '\033[94m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

if __name__ == "__main__":
    start_date = '2020-12-24'
    end_date = '2024-12-24'
    stradle_start_date = '2020-12-31'
    stradle_end_date = '2024-12-31'
    
    # Get prices for both stocks
    for ticker1, ticker2, link in pairs:
        stock_a_prices, _ = get_weekly_prices(ticker1, start_date, end_date)
        stock_b_prices, _ = get_weekly_prices(ticker2, stradle_start_date, stradle_end_date)

        if stock_a_prices and stock_b_prices and len(stock_a_prices) == len(stock_b_prices):
            # Calculate correlation
            correlation = np.corrcoef(stock_a_prices, stock_b_prices)[0, 1]
            
            # Create scatter plot
            plt.figure(figsize=(12, 8))
            plt.scatter(stock_a_prices, stock_b_prices, alpha=0.5)
            plt.xlabel(f'{ticker1} Price ($)')
            plt.ylabel(f'{ticker2} Price ($)')
            plt.title(f'Weekly Price Correlation between {ticker1} and {ticker2}\nCorrelation: {correlation:.3f}')
            
            # Add trend line
            z = np.polyfit(stock_a_prices, stock_b_prices, 1)
            p = np.poly1d(z)
            plt.plot(stock_a_prices, p(stock_a_prices), "r--", alpha=0.8)
            
            # Add grid
            plt.grid(True, alpha=0.3)
            
            # Display plot
            plt.tight_layout()
            print(f"Correlation between {ticker1} and {ticker2}: {correlation:.3f}", end='')
            if len(link) > 0:
                print(f", Yahoo Finance Link: {Colors.BLUE}{Colors.UNDERLINE}{link}{Colors.END}")
            else:
                print() # No link provided
            plt.show()

            showMore = input("q to quit, any other key to continue: ")
            if showMore == 'q' or showMore == 'Q' or showMore == 'quit' or showMore == 'exit' or showMore == 'x' or showMore == 'X':  
                break
            
        else:
            print(f"Error fetching data for {ticker1} and {ticker2} or length mismatch")
