import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import time
import csv
from scipy import stats
from itertools import combinations

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'SNDL', 'MNKD', 'HEXO', 
        'ACB', 'CRON', 'TLRY', 'PLUG', 'FCEL', 'SENS', 'NKLA', 'NIO', 'XPEV', 'BNGO',
        'NAKD', 'OCGN', 'IDEX', 'GNUS', 'CIDM', 'ZOM', 'SOLO', 'ONTX', 'RIOT', 'MARA',
        'CLOV', 'WISH', 'BB', 'NOK', 'EXPR', 'KOSS', 'AMC', 'GME', 'BBBY', 'MVIS',
        'WKHS', 'RIDE', 'FSR', 'HYLN', 'GOEV', 'CHPT', 'BLNK', 'SPCE', 'RMO', 'QS',
        'TSM', 'AMD', 'INTC', 'ASML', 'LRCX', 'AMAT', 'ON', 'QCOM', 'AVGO', 'MU',
        'STM', 'BA', 'GE', 'CAT', 'F', 'GM', 'RIVN', 'LCID', 'NXPI', 'TXN',
        'PYPL', 'SQ', 'SHOP', 'JD', 'BABA', 'TS', 'VALE', 'FCX', 'CCJ',
        'SONY', 'SMCI', 'UMC', 'SMIC', 'GLW', 'KLAC', 'TER', 'ADI', 'MCHP', 'WOLF',
        'HPQ', 'DELL', 'WDC', 'STX', 'SNPS', 'CDNS', 'ARM', 'CY', 'KEYS', 'FLEX',
        'HON', 'MMM', 'ITW', 'EMR', 'ETN', 'PH', 'ROK', 'ABB', 'SIE', 'FANUY',
        'PCAR', 'CMI', 'TEL', 'APH', 'APTV', 'LEA', 'MGA', 'BWA', 'ALV', 'AXL',
        'LIN', 'APD', 'DD', 'DOW', 'PPG', 'ECL', 'ALB', 'CTVA', 'FMC', 'SMG',
        'MT', 'PKX', 'NUE', 'X', 'CLF', 'STLD', 'RS', 'CRS', 'ATI', 'KALU',
        'RIO', 'BHP', 'AA', 'CENX', 'ACH', 'TECK', 'HBM', 'SCCO', 'FM', 'CS',
        'LYB', 'EMN', 'CE', 'OLN', 'KRO', 'MEOH', 'WLK', 'ASH', 'FOE', 'NEU',
        'PANW', 'FTNT', 'CRWD', 'MIME', 'FEYE', 'RPD', 'TENB', 'OKTA', 'CYBR', 'ZS',
        'LTHM', 'SQM', 'LAC', 'PLL', 'LIACF', 'GNENF', 'MALRF', 'NMKEF', 'GALXF', 'PILBF',
        'TMHC', 'LEN', 'PHM', 'DHI', 'KBH', 'TOL', 'BZH', 'MDC', 'MHO', 'HOV',
        'KR', 'WMT', 'TGT', 'COST', 'DG', 'DLTR', 'BJ', 'WBA', 'CVS', 'RAD',
        # Financial Sector
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'BLK',
        # Healthcare
        'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'DHR', 'ABT', 'BMY', 'UNH',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX', 'VLO', 'OXY',
        # Real Estate
        'PLD', 'AMT', 'EQIX', 'CCI', 'DLR', 'O', 'WELL', 'AVB', 'EQR', 'SPG',
        # Entertainment/Streaming
        'NFLX', 'DIS', 'CMCSA', 'PARA', 'WBD', 'SPOT', 'ROKU', 'FUBO', 'LYV', 'IMAX',
        # Airlines
        'DAL', 'UAL', 'AAL', 'LUV', 'JBLU', 'ALK', 'SAVE', 'HA', 'MESA', 'ZCH'
    ]

ticker_data = {}

valid_tickers = []

last_req = 0

def get_weekly_prices(ticker: str, start_date: str, end_date: str) -> list:
    """
    Get weekly stock prices and returns for a given ticker and date range.
    """
    global last_req

    if time.time() - last_req < 0.05:
        time.sleep(0.05)
    
    last_req = time.time()

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1wk')
        
        if df.empty or len(df) < 26:  # Minimum 26 weeks of data
            return None
        
        prices = df['Close'].tolist()
        return prices
    
    except Exception as e:
        return None
    
def find_valid_tickers():
    for ticker in ticker_data.keys():
        if ticker_data[ticker] is not None:
            valid_tickers.append(ticker)

def process_pairs(max_stradle_weeks, output_file):
    count = 0
    top_correlations = []  
    top_r_squared = []    
    top_significant = [] 

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Ticker A', 'Ticker B', 'Weeks', 'Correlation', 'R²', 'P-value'])

    find_valid_tickers()
    print(f"Found {len(valid_tickers)} valid tickers, comapred to {len(ticker_data)} total tickers.")


    for i in range(len(valid_tickers)):
        for j in range(len(valid_tickers)):
            ticker_a = valid_tickers[i]
            ticker_b = valid_tickers[j]

            if i == j or ticker_data[ticker_a] is None or ticker_data[ticker_b] is None:
                break

            for k in range(max_stradle_weeks + 1):
                stock_a_prices = ticker_data[valid_tickers[i]][0]

                stock_a_prices = ticker_data[ticker_a][0] # get non stradled prices

                stock_b_prices = ticker_data[ticker_b][k] # get stradled prices
                
                if len(stock_a_prices) == len(stock_b_prices):
                    try:
                        corr = np.corrcoef(stock_a_prices, stock_b_prices)[0, 1]
                        if not np.isnan(corr):
                            r_squared = corr ** 2
                            n = len(stock_a_prices)
                            t_stat = corr * np.sqrt((n-2)/(1-corr**2))
                            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
                            
                            stats_tuple = (ticker_a, ticker_b, k, corr, r_squared, p_value)
                            
                            with open(output_file, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(stats_tuple)
                            
                            top_correlations.append(stats_tuple)
                            top_correlations.sort(key=lambda x: abs(x[3]), reverse=True)
                            top_correlations = top_correlations[:20]
                            
                            top_r_squared.append(stats_tuple)
                            top_r_squared.sort(key=lambda x: x[4], reverse=True)
                            top_r_squared = top_r_squared[:20]
                            
                            top_significant.append(stats_tuple) 
                            top_significant.sort(key=lambda x: x[5])
                            top_significant = top_significant[:20]
                        
                        count += 1
                        if count % 50 == 0:
                            print(f"Processed {count} pairs")
                            
                    except Exception as e:
                        print(f"Error calculating correlation for {ticker_a} and {ticker_b}: {str(e)}")
                else:
                    break

    with open(output_file, mode='r') as file:
        existing_content = file.read()

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(['Top 20 Strongest Correlations (Absolute)'])
        writer.writerow(['Ticker A', 'Ticker B', 'Weeks', 'Correlation', 'R²', 'P-value'])
        for pair in top_correlations:
            writer.writerow(pair)
            print(f"{pair[0]} - {pair[1]}: Weeks ({pair[2]}), r={pair[3]:.3f}, R²={pair[4]:.3f}, p={pair[5]:.3e}")
        
        writer.writerow([])
        writer.writerow(['Top 20 Highest R² Values'])
        writer.writerow(['Ticker A', 'Ticker B', 'Weeks', 'Correlation', 'R²', 'P-value'])
        for pair in top_r_squared:
            writer.writerow(pair)
        
        writer.writerow([])
        writer.writerow(['Top 20 Most Statistically Significant Pairs'])
        writer.writerow(['Ticker A', 'Ticker B', 'Weeks', 'Correlation', 'R²', 'P-value'])
        for pair in top_significant:
            writer.writerow(pair)
            
        writer.writerow([])
        writer.writerow(['All correlations'])
        writer.writerow(['Ticker A', 'Ticker B', 'Weeks', 'Correlation', 'R²', 'P-value'])
        file.write(existing_content)

def populate_ticker_data(start_date, end_date, max_stradle_weeks):
    for ticker in tickers:
        ticker_data[ticker] = []  # Initialize as empty list first
        for k in range(max_stradle_weeks + 1):
            stock_prices = get_weekly_prices(ticker, datetime.strptime(start_date, '%Y-%m-%d') + timedelta(weeks=k), datetime.strptime(end_date, '%Y-%m-%d') + timedelta(weeks=k))
            if stock_prices:
                ticker_data[ticker].append(stock_prices)
            else:
                ticker_data[ticker] = None
                break

        print(f"Processed {ticker}, {len(ticker_data)} of {len(tickers)}")

if __name__ == "__main__":

    stadle_weeks = input("Stradle weeks? --> ")
    try:
        stadle_weeks = int(stadle_weeks)
    except ValueError:
        print("Stradle weeks must be a number.")
        exit()
    start_date = '2020-11-30'
    end_date = '2024-11-30'

    if datetime.strptime(start_date, '%Y-%m-%d') + timedelta(weeks=stadle_weeks) > datetime.now() - timedelta(weeks=1):
        print(f"End date cannot be within 1 week of today. Please adjust the end date / Max max_stradle_weeks.")
        exit()
    
    populate_ticker_data(start_date, end_date, stadle_weeks)

    output_file = '/home/benjamin-funk/Desktop/stock correlation/waitingLongCorrelations.csv'
    
    process_pairs(stadle_weeks, output_file)
    print(f"\nCorrelations have been written to {output_file}")