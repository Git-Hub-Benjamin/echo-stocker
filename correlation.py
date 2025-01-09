import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

def get_weekly_prices(ticker: str, start_date: str, end_date: str) -> tuple:
    print(f"Fetching data for {ticker} from {start_date} to {end_date}")
    """
    Get weekly stock prices and returns for a given ticker and date range.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1wk')
        
        if df.empty or len(df) < 26:  # Minimum 26 weeks of data
            return None, None
        
        prices = df['Close'].tolist()
        returns = df['Close'].pct_change().dropna().tolist()
        
        return prices, returns
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None, None

def calculate_statistics(stock_a_prices, stock_b_prices, stock_a_returns, stock_b_returns):
    """Calculate comprehensive statistics for a pair of stocks"""
    print(f"Calculating statistics for arrays of lengths: {len(stock_a_prices)}, {len(stock_b_prices)}, {len(stock_a_returns)}, {len(stock_b_returns)}")
    """
    Calculate comprehensive statistics for a pair of stocks
    """
    try:
        # Basic correlation and R-squared
        correlation = np.corrcoef(stock_a_prices, stock_b_prices)[0, 1]
        r_squared = r2_score(stock_b_prices, np.poly1d(np.polyfit(stock_a_prices, stock_b_prices, 1))(stock_a_prices))
        
        # Calculate returns-based statistics
        returns_correlation = np.corrcoef(stock_a_returns[:-1], stock_b_returns[1:])[0, 1]
        
        # Regression analysis on returns
        slope, intercept, r_value, p_value, std_err = stats.linregress(stock_a_returns[:-1], stock_b_returns[1:])
        returns_r_squared = r_value ** 2
        
        # Calculate volatility
        vol_a = np.std(stock_a_returns) * np.sqrt(52)  # Annualized volatility
        vol_b = np.std(stock_b_returns) * np.sqrt(52)
        
        # Calculate beta (using returns)
        beta = np.cov(stock_a_returns[:-1], stock_b_returns[1:])[0][1] / np.var(stock_a_returns[:-1])
        
        # Information coefficient (IC) - correlation of returns
        ic = stats.spearmanr(stock_a_returns[:-1], stock_b_returns[1:])[0]
        
        # Calculate tracking error
        tracking_error = np.std(np.array(stock_b_returns[1:]) - np.array(stock_a_returns[:-1])) * np.sqrt(52)
        
        return {
            'price_correlation': correlation,
            'price_r_squared': r_squared,
            'returns_correlation': returns_correlation,
            'returns_r_squared': returns_r_squared,
            'beta': beta,
            'vol_predictor': vol_a,
            'vol_target': vol_b,
            'tracking_error': tracking_error,
            'information_coefficient': ic,
            'slope': slope,
            'intercept': intercept,
            'p_value': p_value,
            'std_err': std_err
        }
    except Exception as e:
        print(f"Error calculating statistics: {str(e)}")
        return None

def plot_pair_analysis(ticker_a, ticker_b, stock_a_prices, stock_b_prices, stock_a_returns, stock_b_returns, stats_dict):
    """
    Create comprehensive plots for a stock pair
    """
    plt.figure(figsize=(15, 10))
    
    # Returns scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(stock_a_returns[:-1], stock_b_returns[1:], alpha=0.5)
    plt.xlabel(f'{ticker_a} Returns (t)')
    plt.ylabel(f'{ticker_b} Returns (t+1)')
    plt.title(f'Weekly Returns Relationship\nCorr: {stats_dict["returns_correlation"]:.3f}, R²: {stats_dict["returns_r_squared"]:.3f}')
    
    # Add trend line
    z = np.polyfit(stock_a_returns[:-1], stock_b_returns[1:], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(stock_a_returns[:-1]), max(stock_a_returns[:-1]), 100)
    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8)
    
    # Price movement comparison
    plt.subplot(2, 2, 2)
    plt.plot(range(len(stock_a_prices)), stock_a_prices, label=ticker_a)
    plt.plot(range(len(stock_b_prices)), stock_b_prices, label=ticker_b)
    plt.title('Price Movement Comparison')
    plt.legend()
    
    # Returns distribution
    plt.subplot(2, 2, 3)
    sns.kdeplot(data=stock_a_returns[:-1], label=f'{ticker_a} (t)', alpha=0.5)
    sns.kdeplot(data=stock_b_returns[1:], label=f'{ticker_b} (t+1)', alpha=0.5)
    plt.title('Returns Distribution')
    plt.legend()
    
    # Rolling correlation
    plt.subplot(2, 2, 4)
    window = 12  # 12-week rolling correlation
    rolling_corr = [np.corrcoef(stock_a_returns[i:i+window], stock_b_returns[i+1:i+1+window])[0,1] 
                   for i in range(len(stock_a_returns)-window)]
    plt.plot(range(window, len(stock_a_returns)), rolling_corr)
    plt.title('12-Week Rolling Correlation')
    
    plt.tight_layout()
    return plt.gcf()

def main():
    # Initialize tickers list
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'SNDL', 'MNKD', 'HEXO', 
            'ACB', 'CRON', 'TLRY', 'PLUG', 'FCEL', 'SENS', 'NKLA', 'NIO', 'XPEV', 'BNGO',
            'NAKD', 'OCGN', 'IDEX', 'GNUS', 'CIDM', 'ZOM', 'SOLO', 'ONTX', 'RIOT', 'MARA',
            'CLOV', 'WISH', 'BB', 'NOK', 'EXPR', 'KOSS', 'AMC', 'GME', 'MVIS']  # Removed BBBY as it's delisted
    
    # Initialize results storage
    results = []
    
    # Set date ranges (one year of historical data from 2023)
    start_date_a = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    start_date_b = start_date_a + timedelta(weeks=1)  # One week offset
    
    print(f"Analysis period for predictor stocks: {start_date_a.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Analysis period for target stocks: {start_date_b.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("\nAnalyzing stock pairs...")
    
    total_pairs = (len(tickers) * (len(tickers) - 1)) // 2
    pair_count = 0
    valid_pairs = 0
    
    for outer in range(len(tickers)):
        for inner in range(outer + 1, len(tickers)):
            pair_count += 1
            if pair_count % 10 == 0:
                print(f"Progress: {pair_count}/{total_pairs} pairs analyzed ({valid_pairs} valid pairs found)")
            
            # Get data for both stocks
            stock_a_prices, stock_a_returns = get_weekly_prices(tickers[outer], 
                                                              start_date_a.strftime('%Y-%m-%d'), 
                                                              end_date.strftime('%Y-%m-%d'))
            stock_b_prices, stock_b_returns = get_weekly_prices(tickers[inner], 
                                                              start_date_b.strftime('%Y-%m-%d'), 
                                                              end_date.strftime('%Y-%m-%d'))
            
            if all(x is not None for x in [stock_a_prices, stock_b_prices, stock_a_returns, stock_b_returns]):
                if len(stock_a_prices) == len(stock_b_prices) and len(stock_a_prices) > 26:  # Minimum 26 weeks of data
                    # Calculate statistics
                    stats_dict = calculate_statistics(stock_a_prices, stock_b_prices, 
                                                    stock_a_returns, stock_b_returns)
                    
                    if stats_dict is not None:
                        result = {
                            'predictor': tickers[outer],
                            'target': tickers[inner],
                            **stats_dict
                        }
                        results.append(result)
                        valid_pairs += 1
                        print(f"\nValid pair found: {tickers[outer]} -> {tickers[inner]}")
                        print(f"Returns correlation: {stats_dict['returns_correlation']:.3f}")
                        print(f"R-squared: {stats_dict['returns_r_squared']:.3f}")
    
    # Convert to DataFrame and sort by correlation
    if not results:
        print("\nNo valid results found to analyze")
        return
        
    results_df = pd.DataFrame(results)
    
    # Check if we have the required column before sorting
    if 'returns_correlation' in results_df.columns:
        results_df = results_df.sort_values('returns_correlation', ascending=False)
        
        # Print top results
        print("\nTop 10 correlated pairs (based on returns):")
        print("\nTop pairs by absolute correlation:")
        print(results_df[['predictor', 'target', 'returns_correlation', 'returns_r_squared', 
                         'beta', 'p_value']].head(10).to_string())
        
        # Save results to CSV with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f'stock_pair_analysis_{timestamp}.csv'
        results_df.to_csv(csv_filename, index=False)
        print(f"\nFull results saved to: {csv_filename}")
        
        # Plot top 5 pairs
        print("\nGenerating plots for top 5 pairs...")
        for idx, row in results_df.head(5).iterrows():
            stock_a_prices, stock_a_returns = get_weekly_prices(row['predictor'], 
                                                              start_date_a.strftime('%Y-%m-%d'), 
                                                              end_date.strftime('%Y-%m-%d'))
            stock_b_prices, stock_b_returns = get_weekly_prices(row['target'], 
                                                              start_date_b.strftime('%Y-%m-%d'), 
                                                              end_date.strftime('%Y-%m-%d'))
            
            if all(x is not None for x in [stock_a_prices, stock_b_prices, stock_a_returns, stock_b_returns]):
                fig = plot_pair_analysis(row['predictor'], row['target'], 
                                       stock_a_prices, stock_b_prices,
                                       stock_a_returns, stock_b_returns,
                                       row)
                plot_filename = f'pair_analysis_{row["predictor"]}_{row["target"]}_{timestamp}.png'
                fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Plot saved: {plot_filename}")
    else:
        print("Warning: No correlation data available for sorting")
    
    # Save results to CSV
    results_df.to_csv('stock_pair_analysis.csv', index=False)
    
    # Print top results
    print("\nTop 10 correlated pairs (based on returns):")
    print(results_df[['predictor', 'target', 'returns_correlation', 'returns_r_squared', 
                      'beta', 'p_value']].head(10).to_string())
    
    # Plot top 5 pairs
    print("\nGenerating plots for top 5 pairs...")
    for idx, row in results_df.head(5).iterrows():
        stock_a_prices, stock_a_returns = get_weekly_prices(row['predictor'], 
                                                          start_date_a.strftime('%Y-%m-%d'), 
                                                          end_date.strftime('%Y-%m-%d'))
        stock_b_prices, stock_b_returns = get_weekly_prices(row['target'], 
                                                          start_date_b.strftime('%Y-%m-%d'), 
                                                          end_date.strftime('%Y-%m-%d'))
        
        fig = plot_pair_analysis(row['predictor'], row['target'], 
                               stock_a_prices, stock_b_prices,
                               stock_a_returns, stock_b_returns,
                               row)
        fig.savefig(f'pair_analysis_{row["predictor"]}_{row["target"]}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    main()
    
#tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'SNDL', 'MNKD', 'HEXO', 
    # 'ACB', 'CRON', 'TLRY', 'PLUG', 'FCEL', 'SENS', 'NKLA', 'NIO', 'XPEV', 'BNGO',
    # 'NAKD', 'OCGN', 'IDEX', 'GNUS', 'CIDM', 'ZOM', 'SOLO', 'ONTX', 'RIOT', 'MARA',
    # 'CLOV', 'WISH', 'BB', 'NOK', 'EXPR', 'KOSS', 'AMC', 'GME', 'BBBY', 'MVIS',
    # 'WKHS', 'RIDE', 'FSR', 'HYLN', 'GOEV', 'CHPT', 'BLNK', 'SPCE', 'RMO', 'QS',
    # 'TSM', 'AMD', 'INTC', 'ASML', 'LRCX', 'AMAT', 'ON', 'QCOM', 'AVGO', 'MU',
    # 'STM', 'BA', 'GE', 'CAT', 'F', 'GM', 'RIVN', 'LCID', 'NXPI', 'TXN',
    # 'PYPL', 'SQ', 'SHOP', 'JD', 'BABA', 'TS', 'VALE', 'FCX', 'CCJ',
    # 'SONY', 'SMCI', 'UMC', 'SMIC', 'GLW', 'KLAC', 'TER', 'ADI', 'MCHP', 'WOLF',
    # 'HPQ', 'DELL', 'WDC', 'STX', 'SNPS', 'CDNS', 'ARM', 'CY', 'KEYS', 'FLEX',
    # 'HON', 'MMM', 'ITW', 'EMR', 'ETN', 'PH', 'ROK', 'ABB', 'SIE', 'FANUY',
    # 'PCAR', 'CMI', 'TEL', 'APH', 'APTV', 'LEA', 'MGA', 'BWA', 'ALV', 'AXL',
    # 'LIN', 'APD', 'DD', 'DOW', 'PPG', 'ECL', 'ALB', 'CTVA', 'FMC', 'SMG',
    # 'MT', 'PKX', 'NUE', 'X', 'CLF', 'STLD', 'RS', 'CRS', 'ATI', 'KALU',
    # 'RIO', 'BHP', 'AA', 'CENX', 'ACH', 'TECK', 'HBM', 'SCCO', 'FM', 'CS',
    # 'LYB', 'EMN', 'CE', 'OLN', 'KRO', 'MEOH', 'WLK', 'ASH', 'FOE', 'NEU',
    # 'PANW', 'FTNT', 'CRWD', 'MIME', 'FEYE', 'RPD', 'TENB', 'OKTA', 'CYBR', 'ZS',
    # 'LTHM', 'SQM', 'LAC', 'PLL', 'LIACF', 'GNENF', 'MALRF', 'NMKEF', 'GALXF', 'PILBF',
    # 'TMHC', 'LEN', 'PHM', 'DHI', 'KBH', 'TOL', 'BZH', 'MDC', 'MHO', 'HOV',
    # 'KR', 'WMT', 'TGT', 'COST', 'DG', 'DLTR', 'BJ', 'WBA', 'CVS', 'RAD']