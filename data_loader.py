"""
Data loader module for Jegadeesh & Titman (1993) momentum strategy replication.
Handles downloading and loading stock data from various sources.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path

try:
    import wrds
    WRDS_AVAILABLE = True
except ImportError:
    WRDS_AVAILABLE = False
    warnings.warn("WRDS not available. Install with: pip install wrds")

import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOGGING_LEVEL))
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Main class for loading stock market data for momentum strategy analysis.
    Supports multiple data sources: WRDS/CRSP, Yahoo Finance, and local CSV files.
    """
    
    def __init__(self, data_source: str = config.DATA_SOURCE):
        """
        Initialize DataLoader with specified data source.
        
        Parameters:
        -----------
        data_source : str
            Data source to use ('wrds', 'yahoo', 'local_csv')
        """
        self.data_source = data_source
        self.db = None
        
        if data_source == 'wrds' and WRDS_AVAILABLE:
            self._connect_wrds()
        
        # Create data directory if it doesn't exist
        Path(config.DATA_DIR).mkdir(exist_ok=True)
    
    def _connect_wrds(self) -> None:
        """Connect to WRDS database."""
        try:
            self.db = wrds.Connection(wrds_username=config.WRDS_USERNAME)
            logger.info("Successfully connected to WRDS")
        except Exception as e:
            logger.error(f"Failed to connect to WRDS: {e}")
            self.db = None
    
    def load_stock_data(self, 
                       start_date: datetime = config.EXTENDED_START_DATE,
                       end_date: datetime = config.END_DATE,
                       save_to_csv: bool = True) -> pd.DataFrame:
        """
        Load stock return and price data.
        
        Parameters:
        -----------
        start_date : datetime
            Start date for data
        end_date : datetime
            End date for data
        save_to_csv : bool
            Whether to save data to CSV file
            
        Returns:
        --------
        pd.DataFrame
            Stock data with columns: ['date', 'permno', 'ticker', 'ret', 'prc', 'shrout', 'exchcd', 'shrcd']
        """
        logger.info(f"Loading stock data from {start_date} to {end_date}")
        
        if self.data_source == 'wrds':
            data = self._load_from_wrds(start_date, end_date)
        elif self.data_source == 'yahoo':
            data = self._load_from_yahoo(start_date, end_date)
        elif self.data_source == 'local_csv':
            data = self._load_from_csv()
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")
        
        if save_to_csv and not data.empty:
            csv_path = Path(config.DATA_DIR) / "stock_data_raw.csv"
            data.to_csv(csv_path, index=False)
            logger.info(f"Saved raw data to {csv_path}")
        
        logger.info(f"Loaded {len(data):,} observations for {data['permno'].nunique():,} stocks")
        return data
    
    def _load_from_wrds(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load data from WRDS/CRSP."""
        if not self.db:
            raise ConnectionError("WRDS connection not established")
        
        # Format dates for SQL query
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Exchange and share code filters
        exchcd_filter = ','.join(map(str, config.INCLUDED_EXCHANGES))
        shrcd_filter = ','.join(map(str, config.INCLUDED_SHARE_CODES))
        
        # SQL query to get monthly stock data
        query = f"""
        SELECT 
            date,
            permno,
            ticker,
            ret,
            abs(prc) as prc,
            shrout,
            exchcd,
            shrcd,
            vol
        FROM crsp.msf m
        LEFT JOIN crsp.msenames n
        ON m.permno = n.permno
        AND m.date BETWEEN n.namedt AND n.nameenddt
        WHERE m.date BETWEEN '{start_str}' AND '{end_str}'
        AND m.exchcd IN ({exchcd_filter})
        AND m.shrcd IN ({shrcd_filter})
        AND m.ret IS NOT NULL
        AND abs(m.prc) >= {config.MIN_PRICE}
        ORDER BY m.permno, m.date
        """
        
        logger.info("Executing CRSP query...")
        data = self.db.raw_sql(query)
        
        # Convert date column to datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Calculate market capitalization (in millions)
        data['market_cap'] = data['prc'] * data['shrout'] / 1000
        
        return data
    
    def _load_from_yahoo(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load data from Yahoo Finance (for demonstration/testing purposes).
        Note: This is not ideal for academic research due to survivorship bias.
        """
        logger.warning("Using Yahoo Finance data - be aware of survivorship bias!")
        
        # Get list of major stocks (this is a simplified approach)
        # In practice, you'd want a comprehensive list of stocks from the period
        tickers = self._get_major_tickers()
        
        all_data = []
        
        for ticker in tickers:
            try:
                # Download monthly data
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date, interval="1mo")
                
                if hist.empty:
                    continue
                
                # Calculate returns
                hist['ret'] = hist['Close'].pct_change()
                
                # Create DataFrame in CRSP-like format
                ticker_data = pd.DataFrame({
                    'date': hist.index,
                    'permno': hash(ticker) % 10000,  # Fake permno
                    'ticker': ticker,
                    'ret': hist['ret'],
                    'prc': hist['Close'],
                    'shrout': np.nan,  # Not available from Yahoo
                    'exchcd': 1,  # Assume NYSE
                    'shrcd': 10,  # Assume common stock
                    'vol': hist['Volume'],
                    'market_cap': np.nan  # Would need shares outstanding
                })
                
                all_data.append(ticker_data)
                
            except Exception as e:
                logger.warning(f"Failed to download {ticker}: {e}")
                continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _load_from_csv(self) -> pd.DataFrame:
        """Load data from local CSV file."""
        csv_path = Path(config.DATA_DIR) / "stock_data_raw.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        data = pd.read_csv(csv_path)
        data['date'] = pd.to_datetime(data['date'])
        
        logger.info(f"Loaded data from {csv_path}")
        return data
    
    def _get_major_tickers(self) -> List[str]:
        """
        Get list of major stock tickers for Yahoo Finance download.
        This is a simplified list - in practice you'd want comprehensive historical data.
        """
        # Major stocks that were likely around in the 1965-1989 period
        tickers = [
            'AAPL', 'IBM', 'XOM', 'GE', 'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'MRK',
            'VZ', 'T', 'CVX', 'JPM', 'BAC', 'WFC', 'C', 'AXP', 'DIS', 'MMM',
            'CAT', 'BA', 'UTX', 'UNH', 'HD', 'MCD', 'NKE', 'INTC', 'CSCO', 'MSFT',
            'F', 'GM', 'AA', 'AIG', 'GS', 'MS', 'BK', 'HON', 'LMT', 'NOC',
            'DD', 'DOW', 'EMR', 'ITW', 'JCI', 'LLY', 'BMY', 'ABT', 'MDT', 'TMO'
        ]
        return tickers
    
    def load_market_data(self) -> pd.DataFrame:
        """
        Load market portfolio returns and risk-free rates.
        
        Returns:
        --------
        pd.DataFrame
            Market data with columns: ['date', 'market_ret', 'rf_rate']
        """
        logger.info("Loading market and risk-free rate data")
        
        if self.data_source == 'wrds':
            return self._load_market_data_wrds()
        elif self.data_source in ['yahoo', 'local_csv']:
            return self._load_market_data_yahoo()
        else:
            raise ValueError(f"Market data not supported for source: {self.data_source}")
    
    def _load_market_data_wrds(self) -> pd.DataFrame:
        """Load market data from WRDS."""
        if not self.db:
            raise ConnectionError("WRDS connection not established")
        
        start_str = config.EXTENDED_START_DATE.strftime('%Y-%m-%d')
        end_str = config.END_DATE.strftime('%Y-%m-%d')
        
        # Query for market returns (value-weighted CRSP index)
        market_query = f"""
        SELECT 
            date,
            vwretd as market_ret
        FROM crsp.msi
        WHERE date BETWEEN '{start_str}' AND '{end_str}'
        ORDER BY date
        """
        
        market_data = self.db.raw_sql(market_query)
        market_data['date'] = pd.to_datetime(market_data['date'])
        
        # Query for risk-free rates (3-month Treasury bills)
        rf_query = f"""
        SELECT 
            date,
            t30ret as rf_rate
        FROM crsp.msi
        WHERE date BETWEEN '{start_str}' AND '{end_str}'
        ORDER BY date
        """
        
        rf_data = self.db.raw_sql(rf_query)
        rf_data['date'] = pd.to_datetime(rf_data['date'])
        
        # Merge market and risk-free rate data
        market_data = market_data.merge(rf_data, on='date', how='outer')
        
        return market_data
    
    def _load_market_data_yahoo(self) -> pd.DataFrame:
        """Load market data from Yahoo Finance (SPY as proxy)."""
        try:
            # Use SPY as market proxy
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(start=config.EXTENDED_START_DATE, 
                                  end=config.END_DATE, 
                                  interval="1mo")
            
            # Calculate returns
            spy_hist['market_ret'] = spy_hist['Close'].pct_change()
            
            # Create market data DataFrame
            market_data = pd.DataFrame({
                'date': spy_hist.index,
                'market_ret': spy_hist['market_ret'],
                'rf_rate': 0.02 / 12  # Approximate 2% annual risk-free rate
            })
            
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to load market data from Yahoo: {e}")
            return pd.DataFrame()
    
    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        """Save data to CSV file."""
        filepath = Path(config.DATA_DIR) / filename
        data.to_csv(filepath, index=False)
        logger.info(f"Saved data to {filepath}")
    
    def close_connection(self) -> None:
        """Close WRDS connection if open."""
        if self.db:
            self.db.close()
            logger.info("Closed WRDS connection")

def main():
    """Example usage of DataLoader."""
    # Initialize loader
    loader = DataLoader()
    
    # Load stock data
    stock_data = loader.load_stock_data()
    print(f"Loaded {len(stock_data):,} stock observations")
    
    # Load market data
    market_data = loader.load_market_data()
    print(f"Loaded {len(market_data):,} market observations")
    
    # Close connection
    loader.close_connection()

if __name__ == "__main__":
    main()