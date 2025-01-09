import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import time
import csv
from multiprocessing import Pool
from functools import lru_cache
import pandas as pd
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

# Your tickers list stays the same

@lru_cache(maxsize=None)
def get_batch_prices(tickers_tuple, start_date, end_date):
    """Get prices for multiple tickers at once"""
    try:
        time.sleep(0.1)  # Rate limiting
        data = yf.download(list(tickers_tuple), start=start_date, end=end_date, interval='1wk')['Close']
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
        return data
    except Exception as e:
        print(f"Error downloading batch data: {e}")
        return None

def process_chunk(args):
    """Process a chunk of ticker pairs"""
    chunk, start_date, end_date, max_stradle_weeks = args
    results = []
    
    for ticker_a, ticker_b in chunk:
        for k in range(max_stradle_weeks + 1):
            try:
                # Get data for ticker_a (fixed range)
                data_a = get_batch_prices((ticker_a,), start_date, end_date)
                
                # Get data for ticker_b (shifted range)
                shifted_start = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(weeks=k)
                shifted_end = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(weeks=k)
                data_b = get_batch_prices((ticker_b,), shifted_start.strftime('%Y-%m-%d'), 
                                      shifted_end.strftime('%Y-%m-%d'))
                
                if data_a is not None and data_b is not None:
                    common_index = data_a.index.intersection(data_b.index)
                    if len(common_index) >= 26:
                        prices_a = data_a.loc[common_index, ticker_a].values
                        prices_b = data_b.loc[common_index, ticker_b].values
                        
                        corr = np.corrcoef(prices_a, prices_b)[0, 1]
                        if not np.isnan(corr):
                            results.append((ticker_a, ticker_b, k, corr))
            
            except Exception as e:
                print(f"Error processing {ticker_a}-{ticker_b}: {e}")
                continue
                
    return results

def write_results(results, output_file):
    """Write results to CSV file"""
    # Flatten results from all processes
    all_correlations = [item for sublist in results for item in sublist]
    
    # Sort by correlation value to get top 20
    top_correlations = sorted(all_correlations, key=lambda x: x[3], reverse=True)[:20]
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write top 20
        writer.writerow(['Top 20 most correlated pairs'])
        writer.writerow(['Ticker A', 'Ticker B', 'Weeks', 'Correlation'])
        for pair in top_correlations:
            writer.writerow(pair)
            print(f"{pair[0]} - {pair[1]}: Weeks ({pair[2]}), {pair[3]:.3f}")
        
        # Write all correlations
        writer.writerow([])
        writer.writerow(['All correlations'])
        writer.writerow(['Ticker A', 'Ticker B', 'Weeks', 'Correlation'])
        for pair in all_correlations:
            writer.writerow(pair)

def main():
    max_stradle_weeks = 4
    start_date = '2020-11-30'
    end_date = '2024-11-30'
    output_file = '/home/benjamin-funk/Desktop/stock correlation/correlations.csv'

    if datetime.strptime(start_date, '%Y-%m-%d') + timedelta(weeks=max_stradle_weeks) > datetime.now() - timedelta(weeks=1):
        print("End date cannot be within 1 week of today.")
        return

    # Generate all pairs using combinations
    pairs = list(combinations(tickers, 2))

    print(pairs)
    exit()
    
    # Split pairs into chunks for parallel processing
    num_processes = 8  # Adjust based on your CPU
    chunks = np.array_split(pairs, num_processes)
    
    # Prepare arguments for each process
    chunk_args = [(chunk, start_date, end_date, max_stradle_weeks) for chunk in chunks]
    
    # Process chunks in parallel
    with Pool(num_processes) as pool:
        results = pool.map(process_chunk, chunk_args)
    
    # Write results to file
    write_results(results, output_file)
    print(f"\nCorrelations written to {output_file}")

if __name__ == "__main__":
    main()