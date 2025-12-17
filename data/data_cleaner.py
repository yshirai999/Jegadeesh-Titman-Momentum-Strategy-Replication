"""
Data cleaner module for Jegadeesh & Titman (1993) momentum strategy replication.
Handles data preprocessing, cleaning, and filtering operations.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOGGING_LEVEL))
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Class for cleaning and preprocessing stock market data.
    Handles survivorship bias, data quality issues, and applies various filters.
    """
    
    def __init__(self):
        """Initialize DataCleaner."""
        self.cleaning_stats = {
            'initial_observations': 0,
            'final_observations': 0,
            'removed_by_filter': {}
        }
    
    def clean_stock_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main function to clean stock data following academic standards.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw stock data with columns: ['date', 'permno', 'ticker', 'ret', 'prc', 'shrout', 'exchcd', 'shrcd']
        
        Returns:
        --------
        pd.DataFrame
            Cleaned stock data
        """
        logger.info("Starting data cleaning process...")
        
        # Initialize cleaning statistics
        self.cleaning_stats['initial_observations'] = len(data)
        self.cleaning_stats['initial_stocks'] = data['permno'].nunique() if 'permno' in data.columns else data['ticker'].nunique()
        
        # Make a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Step 1: Basic data validation
        cleaned_data = self._validate_basic_data(cleaned_data)
        
        # Step 2: Handle missing values
        cleaned_data = self._handle_missing_values(cleaned_data)
        
        # Step 3: Apply price filters
        cleaned_data = self._apply_price_filters(cleaned_data)
        
        # Step 4: Apply exchange and share code filters
        cleaned_data = self._apply_exchange_share_filters(cleaned_data)
        
        # Step 5: Calculate market capitalization (if not already present)
        cleaned_data = self._calculate_market_cap(cleaned_data)
        
        # Step 6: Apply market cap filters
        cleaned_data = self._apply_market_cap_filters(cleaned_data)
        
        # Step 7: Handle outliers in returns
        cleaned_data = self._handle_return_outliers(cleaned_data)
        
        # Step 8: Ensure minimum data requirements
        cleaned_data = self._apply_minimum_data_requirements(cleaned_data)
        
        # Step 9: Sort data
        cleaned_data = self._sort_data(cleaned_data)
        
        # Final statistics
        self.cleaning_stats['final_observations'] = len(cleaned_data)
        self.cleaning_stats['final_stocks'] = cleaned_data['permno'].nunique() if 'permno' in cleaned_data.columns else cleaned_data['ticker'].nunique()
        
        self._log_cleaning_summary()
        
        return cleaned_data
    
    def _validate_basic_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate basic data structure and types."""
        logger.info("Validating basic data structure...")
        
        # Ensure required columns exist
        required_columns = ['date', 'ret']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert date column to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['ret', 'prc', 'shrout', 'market_cap']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in key variables."""
        logger.info("Handling missing values...")
        
        initial_count = len(data)
        
        # Remove observations with missing returns
        data = data.dropna(subset=['ret'])
        self._update_filter_stats('missing_returns', initial_count - len(data))
        
        # Handle missing prices (if price column exists)
        if 'prc' in data.columns:
            data = data.dropna(subset=['prc'])
            self._update_filter_stats('missing_prices', initial_count - len(data))
        
        return data
    
    def _apply_price_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply minimum price filters to avoid penny stocks."""
        logger.info(f"Applying minimum price filter (${config.MIN_PRICE})...")
        
        if 'prc' not in data.columns:
            logger.warning("Price column not found, skipping price filters")
            return data
        
        initial_count = len(data)
        
        # Filter out stocks below minimum price
        data = data[data['prc'] >= config.MIN_PRICE]
        
        removed_count = initial_count - len(data)
        self._update_filter_stats('min_price', removed_count)
        
        return data
    
    def _apply_exchange_share_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply exchange and share code filters (for CRSP data)."""
        if 'exchcd' not in data.columns or 'shrcd' not in data.columns:
            logger.info("Exchange/share code columns not found, skipping these filters")
            return data
        
        logger.info("Applying exchange and share code filters...")
        
        initial_count = len(data)
        
        # Filter by exchange codes
        data = data[data['exchcd'].isin(config.INCLUDED_EXCHANGES)]
        
        # Filter by share codes  
        data = data[data['shrcd'].isin(config.INCLUDED_SHARE_CODES)]
        
        removed_count = initial_count - len(data)
        self._update_filter_stats('exchange_share_codes', removed_count)
        
        return data
    
    def _calculate_market_cap(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market capitalization if not already present."""
        if 'market_cap' not in data.columns:
            if 'prc' in data.columns and 'shrout' in data.columns:
                logger.info("Calculating market capitalization...")
                # Market cap = Price * Shares Outstanding / 1000 (to get millions)
                data['market_cap'] = data['prc'] * data['shrout'] / 1000
            else:
                logger.warning("Cannot calculate market cap: missing price or shares outstanding")
        
        return data
    
    def _apply_market_cap_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply market capitalization filters."""
        if 'market_cap' not in data.columns:
            logger.warning("Market cap column not found, skipping market cap filters")
            return data
        
        logger.info(f"Applying market cap filter (>{config.MIN_MARKET_CAP_PERCENTILE}th percentile)...")
        
        initial_count = len(data)
        
        # Calculate market cap percentiles by date
        market_cap_percentiles = data.groupby('date')['market_cap'].quantile(
            config.MIN_MARKET_CAP_PERCENTILE / 100
        ).reset_index()
        market_cap_percentiles.columns = ['date', 'min_market_cap']
        
        # Merge with main data
        data = data.merge(market_cap_percentiles, on='date', how='left')
        
        # Apply filter
        data = data[data['market_cap'] >= data['min_market_cap']]
        
        # Drop temporary column
        data = data.drop('min_market_cap', axis=1)
        
        removed_count = initial_count - len(data)
        self._update_filter_stats('market_cap', removed_count)
        
        return data
    
    def _handle_return_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle extreme returns that might be data errors."""
        logger.info("Handling return outliers...")
        
        initial_count = len(data)
        
        # Remove returns greater than 300% or less than -95% (common academic practice)
        # These are likely stock splits, data errors, or other corporate actions
        outlier_mask = (data['ret'] > 3.0) | (data['ret'] < -0.95)
        
        if outlier_mask.sum() > 0:
            logger.info(f"Removing {outlier_mask.sum()} extreme return observations")
            data = data[~outlier_mask]
        
        removed_count = initial_count - len(data)
        self._update_filter_stats('return_outliers', removed_count)
        
        return data
    
    def _apply_minimum_data_requirements(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure stocks have minimum number of observations."""
        logger.info(f"Applying minimum data requirements ({config.MIN_RETURN_OBSERVATIONS} observations)...")
        
        initial_count = len(data)
        
        # Identify stock identifier column
        stock_id = 'permno' if 'permno' in data.columns else 'ticker'
        
        # Count observations per stock
        stock_counts = data.groupby(stock_id).size()
        
        # Keep only stocks with sufficient observations
        valid_stocks = stock_counts[stock_counts >= config.MIN_RETURN_OBSERVATIONS].index
        data = data[data[stock_id].isin(valid_stocks)]
        
        removed_count = initial_count - len(data)
        self._update_filter_stats('min_observations', removed_count)
        
        return data
    
    def _sort_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sort data by stock and date."""
        logger.info("Sorting data...")
        
        # Identify stock identifier column
        stock_id = 'permno' if 'permno' in data.columns else 'ticker'
        
        # Sort by stock ID and date
        data = data.sort_values([stock_id, 'date']).reset_index(drop=True)
        
        return data
    
    def _update_filter_stats(self, filter_name: str, removed_count: int) -> None:
        """Update filtering statistics."""
        self.cleaning_stats['removed_by_filter'][filter_name] = removed_count
        
        if removed_count > 0:
            logger.info(f"  - Removed {removed_count:,} observations ({filter_name})")
    
    def _log_cleaning_summary(self) -> None:
        """Log summary of cleaning process."""
        logger.info("=" * 60)
        logger.info("DATA CLEANING SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Initial observations: {self.cleaning_stats['initial_observations']:,}")
        logger.info(f"Initial stocks: {self.cleaning_stats['initial_stocks']:,}")
        
        logger.info("\nObservations removed by filter:")
        total_removed = 0
        for filter_name, count in self.cleaning_stats['removed_by_filter'].items():
            if count > 0:
                logger.info(f"  - {filter_name}: {count:,}")
                total_removed += count
        
        logger.info(f"\nTotal observations removed: {total_removed:,}")
        logger.info(f"Final observations: {self.cleaning_stats['final_observations']:,}")
        logger.info(f"Final stocks: {self.cleaning_stats['final_stocks']:,}")
        
        retention_rate = (self.cleaning_stats['final_observations'] / 
                         self.cleaning_stats['initial_observations']) * 100
        logger.info(f"Data retention rate: {retention_rate:.1f}%")
        
        logger.info("=" * 60)
    
    def prepare_monthly_panel(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare a balanced monthly panel for momentum analysis.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Cleaned stock data
            
        Returns:
        --------
        pd.DataFrame
            Monthly panel data
        """
        logger.info("Preparing monthly panel data...")
        
        # Identify stock identifier column
        stock_id = 'permno' if 'permno' in data.columns else 'ticker'
        
        # Ensure we have end-of-month dates
        data['year_month'] = data['date'].dt.to_period('M')
        
        # Keep only the last observation per stock per month (in case of duplicates)
        panel_data = (data.groupby([stock_id, 'year_month'])
                          .last()
                          .reset_index())
        
        # Convert period back to timestamp (end of month)
        panel_data['date'] = panel_data['year_month'].dt.to_timestamp('M')
        panel_data = panel_data.drop('year_month', axis=1)
        
        # Ensure data is within our sample period
        panel_data = panel_data[
            (panel_data['date'] >= config.EXTENDED_START_DATE) &
            (panel_data['date'] <= config.END_DATE)
        ]
        
        logger.info(f"Created monthly panel with {len(panel_data):,} observations")
        logger.info(f"Panel covers {panel_data['date'].nunique()} months")
        logger.info(f"Panel includes {panel_data[stock_id].nunique()} unique stocks")
        
        return panel_data
    
    def get_cleaning_stats(self) -> Dict:
        """Return cleaning statistics."""
        return self.cleaning_stats.copy()
    
    def save_cleaned_data(self, data: pd.DataFrame, filename: str = "stock_data_cleaned.csv") -> None:
        """Save cleaned data to CSV."""
        filepath = Path(config.DATA_DIR) / filename
        data.to_csv(filepath, index=False)
        logger.info(f"Saved cleaned data to {filepath}")

def main():
    """Example usage of DataCleaner."""
    from data_loader import DataLoader
    
    # Load raw data
    loader = DataLoader()
    raw_data = loader.load_stock_data()
    loader.close_connection()
    
    # Clean data
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_stock_data(raw_data)
    
    # Prepare monthly panel
    panel_data = cleaner.prepare_monthly_panel(cleaned_data)
    
    # Save cleaned data
    cleaner.save_cleaned_data(panel_data)
    
    # Display cleaning statistics
    stats = cleaner.get_cleaning_stats()
    print("Cleaning completed successfully!")

if __name__ == "__main__":
    main()