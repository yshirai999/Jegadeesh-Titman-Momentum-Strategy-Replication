"""
Data loader module for Jegadeesh & Titman (1993) momentum strategy replication.
Handles downloading and loading stock data from various sources.
"""

import pandas as pd
import numpy as np
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
    Supports WRDS/CRSP and local CSV files.
    """
    
    def __init__(self, data_source: str = config.DATA_SOURCE):
        """
        Initialize DataLoader with specified data source.
        
        Parameters:
        -----------
        data_source : str
            Data source to use ('wrds', 'local_csv')
        """
        self.data_source = data_source
        self.db = None
        
        if data_source == 'wrds' and WRDS_AVAILABLE:
            self._connect_wrds()
        
        # Create data directory if it doesn't exist
        Path(config.DATA_DIR).mkdir(exist_ok=True)
    

    
    def _connect_wrds(self) -> None:
        """Connect to WRDS database with Duo authentication support."""
        try:
            logger.info("Connecting to WRDS (this may require Duo authentication)...")
            
            # For Duo authentication, we need to allow interactive login
            # The wrds library will handle the Duo flow automatically
            if config.WRDS_USERNAME:
                # Use configured username, but allow interactive password/Duo
                print(f"🔐 Connecting to WRDS as user: {config.WRDS_USERNAME}")
                print("📱 You may need to complete Duo authentication...")
                self.db = wrds.Connection(wrds_username=config.WRDS_USERNAME)
            else:
                # Fully interactive login if no username configured
                print("🔐 Interactive WRDS login (will prompt for username and Duo)...")
                self.db = wrds.Connection()
            
            logger.info("✅ Successfully connected to WRDS")
            
        except KeyboardInterrupt:
            logger.warning("WRDS connection cancelled by user")
            self.db = None
        except Exception as e:
            logger.error(f"Failed to connect to WRDS: {e}")
            logger.info("💡 Common Duo authentication issues:")
            logger.info("   - Make sure your institution account is active")
            logger.info("   - Check your Duo Mobile app is working") 
            logger.info("   - Try the web interface first to test credentials")
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
        
        data = pd.DataFrame()  # Initialize empty DataFrame
        
        try:
            # First, try to load from local downloaded CRSP files
            if self.data_source == 'wrds':
                data = self._load_from_wrds(start_date, end_date)
            elif self.data_source == 'local_csv':
                data = self._load_from_csv()
            else:
                raise ValueError(f"Unknown data source: {self.data_source}")
            
            # Check if we got meaningful data
            if data.empty or len(data) == 0:
                raise ValueError(f"No data returned from {self.data_source}")
                
        except Exception as e:
            logger.error(f"Failed to load data from {self.data_source}: {e}")
            raise
        
        if save_to_csv and not data.empty:
            csv_path = Path(config.DATA_DIR) / "stock_data_raw.csv"
            data.to_csv(csv_path, index=False)
            logger.info(f"Saved raw data to {csv_path}")
        
        # Determine actual data source used
        if hasattr(data, '_source_type'):
            actual_source = data._source_type
        else:
            actual_source = self.data_source
            
        logger.info(f"Successfully loaded {len(data):,} observations for {data['permno'].nunique():,} stocks from {actual_source}")
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
            m.date,
            m.permno,
            n.ticker,
            m.ret,
            abs(m.prc) as prc,
            m.shrout,
            n.exchcd,
            n.shrcd,
            m.vol
        FROM crsp.msf m
        LEFT JOIN crsp.msenames n
        ON m.permno = n.permno
        AND m.date BETWEEN n.namedt AND n.nameendt
        WHERE m.date BETWEEN '{start_str}' AND '{end_str}'
        AND n.exchcd IN ({exchcd_filter})
        AND n.shrcd IN ({shrcd_filter})
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
    
    def _load_from_csv(self) -> pd.DataFrame:
        """Load data from local CSV file."""
        csv_path = Path(config.DATA_DIR) / "stock_data_raw.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        data = pd.read_csv(csv_path)
        data['date'] = pd.to_datetime(data['date'])
        
        logger.info(f"Loaded data from {csv_path}")
        return data
    
    def load_market_data(self) -> pd.DataFrame:
        """
        Load market portfolio returns and risk-free rates.
        Uses saved WRDS data if available, otherwise downloads from WRDS.
        
        Returns:
        --------
        pd.DataFrame
            Market data with columns: ['date', 'market_ret', 'rf_rate']
        """
        logger.info("Loading market and risk-free rate data")
        
        # First, try to load from local downloaded files
        market_file = Path(config.DATA_DIR) / "market_data.csv"
        if market_file.exists():
            logger.info(f"Loading market data from {market_file}")
            market_data = pd.read_csv(market_file)
            market_data['date'] = pd.to_datetime(market_data['date'])
            return market_data
        
        # If no local file, try to download from WRDS
        if self.data_source == 'wrds' and self.db:
            return self._load_market_data_wrds(save_to_csv=True)
        else:
            raise FileNotFoundError(
                f"Market data file not found: {market_file}\n"
                "Please download market data first by:\n"
                "1. Setting data_source='wrds' in config\n"
                "2. Running DataLoader with WRDS connection\n"
                "3. Or manually downloading market data from WRDS"
            )
    
    def _load_market_data_wrds(self, save_to_csv: bool = True) -> pd.DataFrame:
        """Load market data from WRDS and optionally save to CSV."""
        if not self.db:
            raise ConnectionError("WRDS connection not established")

        start_str = config.EXTENDED_START_DATE.strftime('%Y-%m-%d')
        end_str = config.END_DATE.strftime('%Y-%m-%d')
        
        logger.info("Downloading market and risk-free rate data from WRDS...")
        
        # Combined query for (monthly) market returns and risk-free rates
        # Average treasury rates when multiple exist for same date
        combined_query = f"""
        SELECT
            m.date AS date,
            m.vwretd AS market_ret,
            AVG(r.tmbidyld * EXTRACT(DAY FROM (m.date + INTERVAL '1 month - 1 day'))) AS rf_rate
        FROM crsp.msi AS m
        LEFT JOIN crsp.tfz_mth_rf2 AS r
            ON m.date = r.mcaldt
        WHERE m.date BETWEEN '{start_str}' AND '{end_str}'
        AND m.vwretd IS NOT NULL
        GROUP BY m.date, m.vwretd
        ORDER BY m.date
        """
        
        market_data = self.db.raw_sql(combined_query)
        market_data['date'] = pd.to_datetime(market_data['date'])
        
        # Handle missing risk-free rate data by forward filling
        market_data['rf_rate'] = market_data['rf_rate'].fillna(method='ffill')
        
        logger.info(f"Downloaded market data: {len(market_data)} observations from {market_data['date'].min().date()} to {market_data['date'].max().date()}")
        
        # Save to CSV if requested
        if save_to_csv:
            csv_path = Path(config.DATA_DIR) / "market_data.csv"
            market_data.to_csv(csv_path, index=False)
            logger.info(f"Saved market data to {csv_path}")

        return market_data

    
    def test_wrds_connection(self) -> bool:
        """Test if WRDS connection is available (may require Duo authentication)."""
        if not WRDS_AVAILABLE:
            logger.warning("WRDS library not installed. Install with: pip install wrds")
            return False
        try:
            if not self.db:
                print("🔍 Testing WRDS connection...")
                self._connect_wrds()
            return self.db is not None
        except KeyboardInterrupt:
            logger.info("WRDS connection test cancelled by user")
            return False
        except Exception as e:
            logger.warning(f"WRDS connection test failed: {e}")
            return False
    
    def load_csv_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            data = pd.read_csv(filepath)
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            return data
        except Exception as e:
            logger.error(f"Error loading CSV file {filepath}: {e}")
            raise

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