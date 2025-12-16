"""
Data validator module for Jegadeesh & Titman (1993) momentum strategy replication.
Validates data quality, completeness, and consistency for momentum analysis.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings

import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOGGING_LEVEL))
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Class for validating stock market data quality and consistency.
    Performs comprehensive checks to ensure data is suitable for momentum analysis.
    """
    
    def __init__(self):
        """Initialize DataValidator."""
        self.validation_results = {
            'passed': [],
            'warnings': [],
            'errors': [],
            'data_quality_score': 0.0
        }
    
    def validate_data(self, data: pd.DataFrame, market_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Comprehensive validation of stock data for momentum analysis.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Stock data to validate
        market_data : pd.DataFrame, optional
            Market and risk-free rate data
            
        Returns:
        --------
        Dict
            Validation results with passed tests, warnings, and errors
        """
        logger.info("Starting comprehensive data validation...")
        
        # Reset validation results
        self.validation_results = {
            'passed': [],
            'warnings': [],
            'errors': [],
            'data_quality_score': 0.0
        }
        
        # Basic structure validation
        self._validate_basic_structure(data)
        
        # Data completeness validation
        self._validate_data_completeness(data)
        
        # Date consistency validation
        self._validate_date_consistency(data)
        
        # Return data validation
        self._validate_return_data(data)
        
        # Price data validation (if available)
        if 'prc' in data.columns:
            self._validate_price_data(data)
        
        # Cross-sectional validation
        self._validate_cross_sectional_data(data)
        
        # Time series validation
        self._validate_time_series_consistency(data)
        
        # Market data validation (if provided)
        if market_data is not None:
            self._validate_market_data(market_data, data)
        
        # Momentum-specific validation
        self._validate_momentum_requirements(data)
        
        # Calculate overall data quality score
        self._calculate_quality_score()
        
        # Log validation summary
        self._log_validation_summary()
        
        return self.validation_results.copy()
    
    def _validate_basic_structure(self, data: pd.DataFrame) -> None:
        """Validate basic data structure and required columns."""
        logger.info("Validating basic data structure...")
        
        # Check if DataFrame is empty
        if data.empty:
            self._add_error("Dataset is empty")
            return
        
        # Check for required columns
        required_columns = ['date', 'ret']
        stock_id_columns = ['permno', 'ticker']
        
        missing_required = [col for col in required_columns if col not in data.columns]
        if missing_required:
            self._add_error(f"Missing required columns: {missing_required}")
        
        # Check for stock identifier
        has_stock_id = any(col in data.columns for col in stock_id_columns)
        if not has_stock_id:
            self._add_error(f"Missing stock identifier columns: {stock_id_columns}")
        
        # Check data types
        if 'date' in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data['date']):
                self._add_warning("Date column is not datetime type")
        
        if 'ret' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['ret']):
                self._add_error("Return column is not numeric type")
        
        if not missing_required and has_stock_id:
            self._add_passed("Basic data structure validation")
    
    def _validate_data_completeness(self, data: pd.DataFrame) -> None:
        """Validate data completeness and coverage."""
        logger.info("Validating data completeness...")
        
        # Check missing values in key columns
        key_columns = ['date', 'ret']
        stock_id = self._get_stock_id_column(data)
        if stock_id:
            key_columns.append(stock_id)
        
        for col in key_columns:
            if col in data.columns:
                missing_count = data[col].isna().sum()
                missing_pct = (missing_count / len(data)) * 100
                
                if missing_pct > 5:  # More than 5% missing
                    self._add_warning(f"High missing data in {col}: {missing_pct:.1f}%")
                elif missing_pct > 0:
                    self._add_warning(f"Missing data in {col}: {missing_count} obs ({missing_pct:.2f}%)")
        
        # Check date coverage
        if 'date' in data.columns and not data['date'].isna().all():
            min_date = data['date'].min()
            max_date = data['date'].max()
            
            expected_start = config.START_DATE
            expected_end = config.END_DATE
            
            if min_date > expected_start:
                self._add_warning(f"Data starts later than expected: {min_date} vs {expected_start}")
            
            if max_date < expected_end:
                self._add_warning(f"Data ends earlier than expected: {max_date} vs {expected_end}")
        
        # Check minimum observations per stock
        if stock_id and 'date' in data.columns:
            stock_counts = data.groupby(stock_id).size()
            insufficient_stocks = (stock_counts < config.MIN_RETURN_OBSERVATIONS).sum()
            
            if insufficient_stocks > 0:
                total_stocks = len(stock_counts)
                self._add_warning(f"{insufficient_stocks}/{total_stocks} stocks have <{config.MIN_RETURN_OBSERVATIONS} observations")
        
        self._add_passed("Data completeness validation")
    
    def _validate_date_consistency(self, data: pd.DataFrame) -> None:
        """Validate date consistency and frequency."""
        logger.info("Validating date consistency...")
        
        if 'date' not in data.columns:
            return
        
        # Check for duplicate dates within stocks
        stock_id = self._get_stock_id_column(data)
        if stock_id:
            duplicates = data.groupby([stock_id, 'date']).size()
            duplicate_count = (duplicates > 1).sum()
            
            if duplicate_count > 0:
                self._add_warning(f"Found {duplicate_count} duplicate date-stock combinations")
        
        # Check date frequency (should be monthly)
        unique_dates = pd.Series(data['date'].dropna().unique()).sort_values()
        if len(unique_dates) > 1:
            date_diffs = unique_dates.diff().dropna()
            
            # Most differences should be around 28-31 days for monthly data
            median_diff = date_diffs.median().days
            
            if not (25 <= median_diff <= 35):
                self._add_warning(f"Unexpected date frequency: median gap = {median_diff} days")
        
        # Check for reasonable date range
        if len(unique_dates) > 0:
            min_date = unique_dates.min()
            max_date = unique_dates.max()
            
            if min_date < pd.Timestamp('1950-01-01'):
                self._add_warning(f"Very early start date: {min_date}")
            
            if max_date > pd.Timestamp('2030-01-01'):
                self._add_warning(f"Future date detected: {max_date}")
        
        self._add_passed("Date consistency validation")
    
    def _validate_return_data(self, data: pd.DataFrame) -> None:
        """Validate return data quality and reasonableness."""
        logger.info("Validating return data...")
        
        if 'ret' not in data.columns:
            return
        
        returns = data['ret'].dropna()
        
        if len(returns) == 0:
            self._add_error("No valid return observations")
            return
        
        # Check for extreme returns
        extreme_positive = (returns > 5.0).sum()  # >500% monthly return
        extreme_negative = (returns < -0.99).sum()  # <-99% monthly return
        
        if extreme_positive > 0:
            self._add_warning(f"Found {extreme_positive} extremely high returns (>500%)")
        
        if extreme_negative > 0:
            self._add_warning(f"Found {extreme_negative} extremely negative returns (<-99%)")
        
        # Check return distribution
        return_stats = {
            'mean': returns.mean(),
            'std': returns.std(),
            'median': returns.median(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
        
        # Reasonable monthly return statistics for US stocks
        if abs(return_stats['mean']) > 0.05:  # |Mean| > 5% monthly
            self._add_warning(f"Unusual mean monthly return: {return_stats['mean']:.3f}")
        
        if return_stats['std'] > 0.5:  # Std > 50% monthly
            self._add_warning(f"Very high return volatility: {return_stats['std']:.3f}")
        
        # Check for constant returns (data error)
        unique_returns = returns.nunique()
        if unique_returns < len(returns) * 0.1:  # <10% unique values
            self._add_warning(f"Low return variability: only {unique_returns} unique values")
        
        self._add_passed("Return data validation")
    
    def _validate_price_data(self, data: pd.DataFrame) -> None:
        """Validate price data (if available)."""
        logger.info("Validating price data...")
        
        if 'prc' not in data.columns:
            return
        
        prices = data['prc'].dropna()
        
        if len(prices) == 0:
            self._add_warning("No valid price observations")
            return
        
        # Check for negative prices (shouldn't happen with abs() in data loading)
        negative_prices = (prices < 0).sum()
        if negative_prices > 0:
            self._add_warning(f"Found {negative_prices} negative prices")
        
        # Check for zero prices
        zero_prices = (prices == 0).sum()
        if zero_prices > 0:
            self._add_warning(f"Found {zero_prices} zero prices")
        
        # Check price reasonableness (assuming USD)
        very_low_prices = (prices < 0.01).sum()
        very_high_prices = (prices > 10000).sum()
        
        if very_low_prices > len(prices) * 0.01:  # >1% of observations
            self._add_warning(f"Many very low prices (<$0.01): {very_low_prices}")
        
        if very_high_prices > 0:
            self._add_warning(f"Very high prices (>$10,000): {very_high_prices}")
        
        # Validate price-return consistency (if both available)
        if 'ret' in data.columns:
            self._validate_price_return_consistency(data)
        
        self._add_passed("Price data validation")
    
    def _validate_price_return_consistency(self, data: pd.DataFrame) -> None:
        """Validate consistency between prices and returns."""
        logger.info("Validating price-return consistency...")
        
        stock_id = self._get_stock_id_column(data)
        if not stock_id:
            return
        
        inconsistencies = 0
        total_checks = 0
        
        for stock in data[stock_id].unique()[:100]:  # Check sample of stocks
            stock_data = data[data[stock_id] == stock].sort_values('date')
            
            if len(stock_data) < 2:
                continue
            
            # Calculate returns from prices
            price_returns = stock_data['prc'].pct_change().dropna()
            reported_returns = stock_data['ret'].iloc[1:len(price_returns)+1]
            
            # Compare (allowing for some tolerance due to dividends, etc.)
            if len(price_returns) > 0 and len(reported_returns) > 0:
                min_len = min(len(price_returns), len(reported_returns))
                price_returns = price_returns.iloc[:min_len]
                reported_returns = reported_returns.iloc[:min_len].values
                
                differences = abs(price_returns - reported_returns)
                large_diffs = (differences > config.PRICE_RETURN_TOLERANCE).sum()
                
                inconsistencies += large_diffs
                total_checks += len(differences)
        
        if total_checks > 0:
            inconsistency_rate = inconsistencies / total_checks
            if inconsistency_rate > 0.1:  # >10% inconsistencies
                self._add_warning(f"Price-return inconsistency rate: {inconsistency_rate:.2f}")
    
    def _validate_cross_sectional_data(self, data: pd.DataFrame) -> None:
        """Validate cross-sectional aspects of the data."""
        logger.info("Validating cross-sectional data...")
        
        stock_id = self._get_stock_id_column(data)
        if not stock_id:
            return
        
        # Check number of stocks per month
        monthly_stock_counts = data.groupby('date')[stock_id].nunique()
        
        min_stocks = monthly_stock_counts.min()
        mean_stocks = monthly_stock_counts.mean()
        
        if min_stocks < config.MIN_STOCKS_PER_MONTH:
            self._add_warning(f"Some months have few stocks: minimum = {min_stocks}")
        
        # Check for reasonable variation in stock counts over time
        stock_count_cv = monthly_stock_counts.std() / monthly_stock_counts.mean()
        if stock_count_cv > 0.5:  # Coefficient of variation > 50%
            self._add_warning(f"High variation in monthly stock counts: CV = {stock_count_cv:.2f}")
        
        # Check market cap distribution (if available)
        if 'market_cap' in data.columns:
            self._validate_market_cap_distribution(data)
        
        self._add_passed("Cross-sectional data validation")
    
    def _validate_market_cap_distribution(self, data: pd.DataFrame) -> None:
        """Validate market capitalization distribution."""
        market_caps = data['market_cap'].dropna()
        
        if len(market_caps) == 0:
            return
        
        # Check for reasonable market cap range
        median_market_cap = market_caps.median()
        
        # Very rough check: median market cap should be reasonable
        if median_market_cap < 1:  # <$1M median market cap seems low
            self._add_warning(f"Low median market cap: ${median_market_cap:.1f}M")
        
        if median_market_cap > 100000:  # >$100B median seems high for historical data
            self._add_warning(f"High median market cap: ${median_market_cap:.1f}M")
    
    def _validate_time_series_consistency(self, data: pd.DataFrame) -> None:
        """Validate time series consistency within stocks."""
        logger.info("Validating time series consistency...")
        
        stock_id = self._get_stock_id_column(data)
        if not stock_id:
            return
        
        # Check for reasonable time series length distribution
        stock_lengths = data.groupby(stock_id).size()
        
        very_short_series = (stock_lengths < 12).sum()  # <1 year
        very_long_series = (stock_lengths > 300).sum()   # >25 years
        
        total_stocks = len(stock_lengths)
        
        if very_short_series > total_stocks * 0.5:  # >50% very short
            self._add_warning(f"Many short time series: {very_short_series}/{total_stocks}")
        
        # Check for gaps in time series
        gaps_detected = 0
        for stock in data[stock_id].unique()[:100]:  # Sample
            stock_data = data[data[stock_id] == stock].sort_values('date')
            if len(stock_data) > 1:
                date_diffs = stock_data['date'].diff().dt.days.dropna()
                # Gap if difference > 45 days (more than 1.5 months)
                if (date_diffs > 45).any():
                    gaps_detected += 1
        
        if gaps_detected > 10:
            self._add_warning(f"Time series gaps detected in {gaps_detected} stocks")
        
        self._add_passed("Time series consistency validation")
    
    def _validate_market_data(self, market_data: pd.DataFrame, stock_data: pd.DataFrame) -> None:
        """Validate market and risk-free rate data."""
        logger.info("Validating market data...")
        
        if market_data.empty:
            self._add_error("Market data is empty")
            return
        
        # Check required columns
        if 'market_ret' not in market_data.columns:
            self._add_error("Missing market return column")
        
        if 'rf_rate' not in market_data.columns:
            self._add_warning("Missing risk-free rate column")
        
        # Check date alignment with stock data
        stock_dates = set(stock_data['date'].dropna())
        market_dates = set(market_data['date'].dropna())
        
        missing_market_dates = stock_dates - market_dates
        if missing_market_dates:
            self._add_warning(f"Market data missing for {len(missing_market_dates)} stock data dates")
        
        # Validate market return reasonableness
        if 'market_ret' in market_data.columns:
            market_rets = market_data['market_ret'].dropna()
            
            if len(market_rets) > 0:
                # Reasonable monthly market return statistics
                mean_market_ret = market_rets.mean()
                std_market_ret = market_rets.std()
                
                if abs(mean_market_ret) > 0.03:  # |Mean| > 3% monthly
                    self._add_warning(f"Unusual mean market return: {mean_market_ret:.3f}")
                
                if std_market_ret > 0.15:  # Std > 15% monthly
                    self._add_warning(f"High market return volatility: {std_market_ret:.3f}")
        
        # Validate risk-free rate
        if 'rf_rate' in market_data.columns:
            rf_rates = market_data['rf_rate'].dropna()
            
            if len(rf_rates) > 0:
                mean_rf = rf_rates.mean()
                
                if mean_rf < 0 or mean_rf > 0.02:  # Outside 0-24% annual
                    self._add_warning(f"Unusual risk-free rate: {mean_rf:.4f} monthly")
        
        self._add_passed("Market data validation")
    
    def _validate_momentum_requirements(self, data: pd.DataFrame) -> None:
        """Validate data meets momentum strategy requirements."""
        logger.info("Validating momentum strategy requirements...")
        
        stock_id = self._get_stock_id_column(data)
        if not stock_id:
            return
        
        # Check we have sufficient history for longest formation period
        max_formation = max(config.FORMATION_PERIODS)
        
        sufficient_history_stocks = 0
        total_stocks = data[stock_id].nunique()
        
        for stock in data[stock_id].unique():
            stock_data = data[data[stock_id] == stock].sort_values('date')
            if len(stock_data) >= max_formation + 1:  # +1 for return calculation
                sufficient_history_stocks += 1
        
        coverage_rate = sufficient_history_stocks / total_stocks
        
        if coverage_rate < 0.5:  # <50% of stocks have sufficient history
            self._add_warning(f"Low momentum coverage: {coverage_rate:.1%} stocks have sufficient history")
        
        # Check temporal coverage for strategy periods
        date_range = data['date'].max() - data['date'].min()
        required_range = timedelta(days=365 * (max(config.FORMATION_PERIODS) + max(config.HOLDING_PERIODS)) / 12)
        
        if date_range < required_range:
            self._add_warning("Limited date range for momentum strategy testing")
        
        self._add_passed("Momentum strategy requirements validation")
    
    def _get_stock_id_column(self, data: pd.DataFrame) -> Optional[str]:
        """Get the stock identifier column name."""
        if 'permno' in data.columns:
            return 'permno'
        elif 'ticker' in data.columns:
            return 'ticker'
        else:
            return None
    
    def _add_passed(self, test_name: str) -> None:
        """Add a passed test to results."""
        self.validation_results['passed'].append(test_name)
    
    def _add_warning(self, message: str) -> None:
        """Add a warning to results."""
        self.validation_results['warnings'].append(message)
        logger.warning(f"VALIDATION WARNING: {message}")
    
    def _add_error(self, message: str) -> None:
        """Add an error to results."""
        self.validation_results['errors'].append(message)
        logger.error(f"VALIDATION ERROR: {message}")
    
    def _calculate_quality_score(self) -> None:
        """Calculate overall data quality score."""
        total_tests = len(self.validation_results['passed']) + len(self.validation_results['warnings']) + len(self.validation_results['errors'])
        
        if total_tests == 0:
            self.validation_results['data_quality_score'] = 0.0
            return
        
        # Score: passed tests get full points, warnings get half points, errors get zero
        score = (len(self.validation_results['passed']) + 
                0.5 * len(self.validation_results['warnings'])) / total_tests
        
        self.validation_results['data_quality_score'] = round(score * 100, 1)
    
    def _log_validation_summary(self) -> None:
        """Log validation summary."""
        logger.info("=" * 60)
        logger.info("DATA VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Tests passed: {len(self.validation_results['passed'])}")
        logger.info(f"Warnings: {len(self.validation_results['warnings'])}")
        logger.info(f"Errors: {len(self.validation_results['errors'])}")
        logger.info(f"Data quality score: {self.validation_results['data_quality_score']}/100")
        
        if self.validation_results['errors']:
            logger.info("\nErrors found:")
            for error in self.validation_results['errors']:
                logger.info(f"  - {error}")
        
        if self.validation_results['warnings']:
            logger.info("\nWarnings:")
            for warning in self.validation_results['warnings'][:10]:  # Show first 10
                logger.info(f"  - {warning}")
            
            if len(self.validation_results['warnings']) > 10:
                logger.info(f"  ... and {len(self.validation_results['warnings']) - 10} more warnings")
        
        logger.info("=" * 60)
    
    def is_data_valid(self) -> bool:
        """Check if data passes validation (no errors)."""
        return len(self.validation_results['errors']) == 0
    
    def get_validation_report(self) -> str:
        """Get formatted validation report."""
        report = []
        report.append("=" * 60)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 60)
        
        report.append(f"Overall Quality Score: {self.validation_results['data_quality_score']}/100")
        report.append("")
        
        if self.validation_results['passed']:
            report.append(f"PASSED TESTS ({len(self.validation_results['passed'])}):")
            for test in self.validation_results['passed']:
                report.append(f"  ✓ {test}")
            report.append("")
        
        if self.validation_results['warnings']:
            report.append(f"WARNINGS ({len(self.validation_results['warnings'])}):")
            for warning in self.validation_results['warnings']:
                report.append(f"  ⚠ {warning}")
            report.append("")
        
        if self.validation_results['errors']:
            report.append(f"ERRORS ({len(self.validation_results['errors'])}):")
            for error in self.validation_results['errors']:
                report.append(f"  ✗ {error}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)

def main():
    """Example usage of DataValidator."""
    from data_loader import DataLoader
    from data_cleaner import DataCleaner
    
    # Load and clean data
    loader = DataLoader()
    raw_data = loader.load_stock_data()
    market_data = loader.load_market_data()
    loader.close_connection()
    
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_stock_data(raw_data)
    
    # Validate data
    validator = DataValidator()
    results = validator.validate_data(cleaned_data, market_data)
    
    # Print report
    print(validator.get_validation_report())
    
    # Save validation report
    report_path = Path(config.DATA_DIR) / "validation_report.txt"
    with open(report_path, 'w') as f:
        f.write(validator.get_validation_report())
    
    print(f"Validation report saved to: {report_path}")

if __name__ == "__main__":
    main()