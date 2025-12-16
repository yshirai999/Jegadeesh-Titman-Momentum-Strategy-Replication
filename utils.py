"""
Utility functions for Jegadeesh & Titman (1993) momentum strategy replication.
Provides helper functions and common utilities used across modules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import os
from pathlib import Path
import logging
from datetime import datetime, date
import warnings

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def ensure_directory_exists(directory_path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Parameters:
    -----------
    directory_path : str
        Path to directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def setup_logging(log_level: str = 'INFO', 
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Parameters:
    -----------
    log_level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    log_file : str, optional
        Path to log file. If None, logs to console only.
        
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        ensure_directory_exists(os.path.dirname(log_file))
        logging.basicConfig(
            level=numeric_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=numeric_level,
            format=log_format,
            handlers=[logging.StreamHandler()]
        )
    
    return logging.getLogger(__name__)

def validate_date_range(start_date: Union[str, datetime, date], 
                       end_date: Union[str, datetime, date]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Validate and convert date range to pandas Timestamps.
    
    Parameters:
    -----------
    start_date : str, datetime, or date
        Start date
    end_date : str, datetime, or date
        End date
        
    Returns:
    --------
    Tuple[pd.Timestamp, pd.Timestamp]
        Validated start and end dates
        
    Raises:
    -------
    ValueError
        If date range is invalid
    """
    # Convert to pandas Timestamps
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    
    # Validate range
    if start_ts >= end_ts:
        raise ValueError(f"Start date ({start_ts}) must be before end date ({end_ts})")
    
    return start_ts, end_ts

def calculate_compound_return(returns: pd.Series) -> float:
    """
    Calculate compound return from a series of returns.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of period returns
        
    Returns:
    --------
    float
        Compound return
    """
    if returns.empty or returns.isna().all():
        return np.nan
    
    # Remove NaN values
    clean_returns = returns.dropna()
    
    if len(clean_returns) == 0:
        return np.nan
    
    # Calculate compound return: (1 + r1) * (1 + r2) * ... - 1
    compound = (1 + clean_returns).prod() - 1
    
    return compound

def annualize_return(return_value: float, periods_per_year: int = 12) -> float:
    """
    Annualize a return value.
    
    Parameters:
    -----------
    return_value : float
        Return value to annualize
    periods_per_year : int
        Number of periods per year (12 for monthly, 252 for daily)
        
    Returns:
    --------
    float
        Annualized return
    """
    if pd.isna(return_value):
        return np.nan
    
    return return_value * periods_per_year

def annualize_volatility(volatility: float, periods_per_year: int = 12) -> float:
    """
    Annualize a volatility measure.
    
    Parameters:
    -----------
    volatility : float
        Volatility to annualize
    periods_per_year : int
        Number of periods per year (12 for monthly, 252 for daily)
        
    Returns:
    --------
    float
        Annualized volatility
    """
    if pd.isna(volatility):
        return np.nan
    
    return volatility * np.sqrt(periods_per_year)

def calculate_sharpe_ratio(returns: pd.Series, 
                          risk_free_rate: float = 0.0,
                          periods_per_year: int = 12) -> float:
    """
    Calculate Sharpe ratio.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Number of periods per year
        
    Returns:
    --------
    float
        Sharpe ratio
    """
    if returns.empty or returns.isna().all():
        return np.nan
    
    # Calculate excess returns
    period_rf = risk_free_rate / periods_per_year
    excess_returns = returns - period_rf
    
    # Calculate Sharpe ratio
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()
    
    if std_excess == 0 or pd.isna(std_excess):
        return np.nan
    
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    
    return sharpe

def calculate_maximum_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from return series.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
        
    Returns:
    --------
    float
        Maximum drawdown (negative value)
    """
    if returns.empty or returns.isna().all():
        return np.nan
    
    # Calculate cumulative returns
    cumulative = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cumulative.expanding().max()
    
    # Calculate drawdowns
    drawdown = (cumulative - running_max) / running_max
    
    # Return maximum drawdown (most negative value)
    return drawdown.min()

def winsorize_data(data: pd.Series, 
                  lower_percentile: float = 0.01, 
                  upper_percentile: float = 0.99) -> pd.Series:
    """
    Winsorize data by capping extreme values at specified percentiles.
    
    Parameters:
    -----------
    data : pd.Series
        Data to winsorize
    lower_percentile : float
        Lower percentile (e.g., 0.01 for 1st percentile)
    upper_percentile : float
        Upper percentile (e.g., 0.99 for 99th percentile)
        
    Returns:
    --------
    pd.Series
        Winsorized data
    """
    if data.empty:
        return data
    
    # Calculate percentile values
    lower_bound = data.quantile(lower_percentile)
    upper_bound = data.quantile(upper_percentile)
    
    # Clip values
    winsorized = data.clip(lower=lower_bound, upper=upper_bound)
    
    return winsorized

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format a decimal value as a percentage string.
    
    Parameters:
    -----------
    value : float
        Decimal value to format (e.g., 0.05 for 5%)
    decimal_places : int
        Number of decimal places
        
    Returns:
    --------
    str
        Formatted percentage string
    """
    if pd.isna(value):
        return "N/A"
    
    percentage = value * 100
    return f"{percentage:.{decimal_places}f}%"

def format_number(value: float, decimal_places: int = 3) -> str:
    """
    Format a number with specified decimal places.
    
    Parameters:
    -----------
    value : float
        Value to format
    decimal_places : int
        Number of decimal places
        
    Returns:
    --------
    str
        Formatted number string
    """
    if pd.isna(value):
        return "N/A"
    
    return f"{value:.{decimal_places}f}"

def create_portfolio_weights(market_caps: pd.Series, 
                           weighting_scheme: str = 'equal') -> pd.Series:
    """
    Create portfolio weights based on specified scheme.
    
    Parameters:
    -----------
    market_caps : pd.Series
        Market capitalization values (for value weighting)
    weighting_scheme : str
        'equal' for equal weighting, 'value' for market cap weighting
        
    Returns:
    --------
    pd.Series
        Portfolio weights
    """
    if market_caps.empty:
        return pd.Series(dtype=float)
    
    if weighting_scheme == 'equal':
        # Equal weights
        n_stocks = len(market_caps)
        weights = pd.Series(1.0 / n_stocks, index=market_caps.index)
    
    elif weighting_scheme == 'value':
        # Market cap weights
        total_market_cap = market_caps.sum()
        if total_market_cap == 0:
            # Fallback to equal weights if no market cap data
            n_stocks = len(market_caps)
            weights = pd.Series(1.0 / n_stocks, index=market_caps.index)
        else:
            weights = market_caps / total_market_cap
    
    else:
        raise ValueError(f"Unknown weighting scheme: {weighting_scheme}")
    
    return weights

def calculate_portfolio_return(returns: pd.Series, weights: pd.Series) -> float:
    """
    Calculate weighted portfolio return.
    
    Parameters:
    -----------
    returns : pd.Series
        Individual asset returns
    weights : pd.Series
        Portfolio weights
        
    Returns:
    --------
    float
        Portfolio return
    """
    # Align indices and handle missing data
    aligned_returns, aligned_weights = returns.align(weights, join='inner')
    
    if aligned_returns.empty or aligned_weights.empty:
        return np.nan
    
    # Calculate weighted return
    portfolio_return = (aligned_returns * aligned_weights).sum()
    
    return portfolio_return

def get_trading_days_in_month(date: pd.Timestamp) -> int:
    """
    Get approximate number of trading days in a month.
    
    Parameters:
    -----------
    date : pd.Timestamp
        Date in the month
        
    Returns:
    --------
    int
        Approximate trading days (assumes ~21 trading days per month)
    """
    # Simple approximation: assume 21 trading days per month
    # This could be made more accurate with a trading calendar
    return 21

def check_data_quality(data: pd.DataFrame, 
                      required_columns: List[str],
                      date_column: str = 'date') -> Dict[str, Any]:
    """
    Check data quality and return summary statistics.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to check
    required_columns : List[str]
        Required column names
    date_column : str
        Name of date column
        
    Returns:
    --------
    Dict[str, Any]
        Data quality summary
    """
    quality_summary = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_columns': [],
        'data_coverage': {},
        'date_range': {},
        'data_types': {},
        'issues': []
    }
    
    # Check for missing required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    quality_summary['missing_columns'] = missing_columns
    
    if missing_columns:
        quality_summary['issues'].append(f"Missing required columns: {missing_columns}")
    
    # Check data coverage
    for column in data.columns:
        total_values = len(data)
        non_null_values = data[column].notna().sum()
        coverage = non_null_values / total_values if total_values > 0 else 0
        quality_summary['data_coverage'][column] = coverage
        
        if coverage < 0.5:  # Less than 50% coverage
            quality_summary['issues'].append(f"Low data coverage in {column}: {coverage:.1%}")
    
    # Check date range
    if date_column in data.columns:
        try:
            date_series = pd.to_datetime(data[date_column])
            quality_summary['date_range'] = {
                'start_date': date_series.min(),
                'end_date': date_series.max(),
                'total_periods': date_series.nunique()
            }
        except Exception as e:
            quality_summary['issues'].append(f"Date column issue: {e}")
    
    # Check data types
    quality_summary['data_types'] = data.dtypes.to_dict()
    
    return quality_summary

def save_results_to_csv(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                       output_path: str,
                       index: bool = True) -> None:
    """
    Save results to CSV file(s).
    
    Parameters:
    -----------
    data : pd.DataFrame or Dict[str, pd.DataFrame]
        Data to save
    output_path : str
        Output file path (for single DataFrame) or directory (for multiple)
    index : bool
        Whether to include index in CSV
    """
    if isinstance(data, pd.DataFrame):
        # Single DataFrame
        ensure_directory_exists(os.path.dirname(output_path))
        data.to_csv(output_path, index=index)
    
    elif isinstance(data, dict):
        # Multiple DataFrames
        ensure_directory_exists(output_path)
        
        for name, df in data.items():
            filename = f"{name}.csv"
            filepath = os.path.join(output_path, filename)
            df.to_csv(filepath, index=index)
    
    else:
        raise ValueError("Data must be DataFrame or dict of DataFrames")

def load_results_from_csv(input_path: str) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load results from CSV file(s).
    
    Parameters:
    -----------
    input_path : str
        Input file path or directory
        
    Returns:
    --------
    pd.DataFrame or Dict[str, pd.DataFrame]
        Loaded data
    """
    if os.path.isfile(input_path):
        # Single file
        return pd.read_csv(input_path, index_col=0)
    
    elif os.path.isdir(input_path):
        # Directory with multiple files
        results = {}
        
        for filename in os.listdir(input_path):
            if filename.endswith('.csv'):
                name = filename[:-4]  # Remove .csv extension
                filepath = os.path.join(input_path, filename)
                results[name] = pd.read_csv(filepath, index_col=0)
        
        return results
    
    else:
        raise ValueError(f"Path does not exist: {input_path}")

def print_summary_statistics(data: pd.DataFrame, 
                           numeric_columns: Optional[List[str]] = None) -> None:
    """
    Print summary statistics for a DataFrame.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to summarize
    numeric_columns : List[str], optional
        Specific columns to summarize. If None, uses all numeric columns.
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nDataFrame shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}" if isinstance(data.index, pd.DatetimeIndex) else "")
    
    if numeric_columns:
        print(f"\nNumeric columns: {len(numeric_columns)}")
        print(data[numeric_columns].describe())
    
    # Missing data summary
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        print("\nMissing data:")
        print(missing_data[missing_data > 0])

def main():
    """
    Example usage of utility functions.
    """
    # Example of setting up logging
    logger = setup_logging('INFO')
    logger.info("Utilities module loaded successfully")
    
    # Example of calculating performance metrics
    returns = pd.Series([0.01, -0.02, 0.03, 0.005, -0.01])
    
    compound_ret = calculate_compound_return(returns)
    sharpe = calculate_sharpe_ratio(returns)
    max_dd = calculate_maximum_drawdown(returns)
    
    print(f"\nExample calculations:")
    print(f"Compound return: {format_percentage(compound_ret)}")
    print(f"Sharpe ratio: {format_number(sharpe)}")
    print(f"Maximum drawdown: {format_percentage(max_dd)}")

if __name__ == "__main__":
    main()
    float
        Volatility
    """
    returns_array = np.array(returns)
    clean_returns = returns_array[~np.isnan(returns_array)]
    
    if len(clean_returns) < 2:
        return np.nan
    
    vol = np.std(clean_returns, ddof=1)
    
    if annualize and frequency == 'monthly':
        vol *= np.sqrt(12)
    elif annualize and frequency == 'daily':
        vol *= np.sqrt(252)
    elif annualize and frequency == 'quarterly':
        vol *= np.sqrt(4)
    
    return vol

def calculate_sharpe_ratio(returns: Union[List, np.ndarray, pd.Series],
                          risk_free_rate: float = 0.0,
                          frequency: str = 'monthly') -> float:
    """
    Calculate Sharpe ratio.
    
    Parameters:
    -----------
    returns : Union[List, np.ndarray, pd.Series]
        Series of returns
    risk_free_rate : float
        Risk-free rate (annualized)
    frequency : str
        Frequency of returns
        
    Returns:
    --------
    float
        Sharpe ratio
    """
    excess_returns = np.array(returns) - risk_free_rate / 12  # Convert to monthly
    
    mean_excess = np.nanmean(excess_returns)
    vol_excess = np.nanstd(excess_returns, ddof=1)
    
    if vol_excess == 0 or np.isnan(vol_excess):
        return np.nan
    
    sharpe = mean_excess / vol_excess
    
    if frequency == 'monthly':
        sharpe *= np.sqrt(12)  # Annualize
    
    return sharpe

def winsorize(data: Union[List, np.ndarray, pd.Series],
              lower_percentile: float = 0.01,
              upper_percentile: float = 0.99) -> np.ndarray:
    """
    Winsorize data by capping extreme values.
    
    Parameters:
    -----------
    data : Union[List, np.ndarray, pd.Series]
        Data to winsorize
    lower_percentile : float
        Lower percentile for winsorization
    upper_percentile : float
        Upper percentile for winsorization
        
    Returns:
    --------
    np.ndarray
        Winsorized data
    """
    data_array = np.array(data)
    
    # Calculate percentiles (ignoring NaN values)
    lower_bound = np.nanpercentile(data_array, lower_percentile * 100)
    upper_bound = np.nanpercentile(data_array, upper_percentile * 100)
    
    # Apply winsorization
    winsorized = np.copy(data_array)
    winsorized[winsorized < lower_bound] = lower_bound
    winsorized[winsorized > upper_bound] = upper_bound
    
    return winsorized

# =============================================================================
# PORTFOLIO UTILITY FUNCTIONS
# =============================================================================

def calculate_portfolio_return(weights: Union[List, np.ndarray, pd.Series],
                             returns: Union[List, np.ndarray, pd.Series]) -> float:
    """
    Calculate weighted portfolio return.
    
    Parameters:
    -----------
    weights : Union[List, np.ndarray, pd.Series]
        Portfolio weights
    returns : Union[List, np.ndarray, pd.Series]
        Individual asset returns
        
    Returns:
    --------
    float
        Portfolio return
    """
    weights_array = np.array(weights)
    returns_array = np.array(returns)
    
    # Handle missing values
    valid_mask = ~(np.isnan(weights_array) | np.isnan(returns_array))
    
    if not np.any(valid_mask):
        return np.nan
    
    valid_weights = weights_array[valid_mask]
    valid_returns = returns_array[valid_mask]
    
    # Renormalize weights
    weight_sum = np.sum(valid_weights)
    if weight_sum == 0:
        return np.nan
    
    normalized_weights = valid_weights / weight_sum
    
    return np.sum(normalized_weights * valid_returns)

def rebalance_portfolio_weights(current_weights: Union[List, np.ndarray, pd.Series],
                               returns: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
    """
    Calculate new portfolio weights after returns (before rebalancing).
    
    Parameters:
    -----------
    current_weights : Union[List, np.ndarray, pd.Series]
        Current portfolio weights
    returns : Union[List, np.ndarray, pd.Series]
        Period returns
        
    Returns:
    --------
    np.ndarray
        New weights after returns
    """
    weights_array = np.array(current_weights)
    returns_array = np.array(returns)
    
    # Calculate new values
    new_values = weights_array * (1 + returns_array)
    
    # Handle missing values
    valid_mask = ~np.isnan(new_values)
    
    if not np.any(valid_mask):
        return weights_array
    
    # Renormalize to sum to 1
    total_value = np.sum(new_values[valid_mask])
    
    if total_value == 0:
        return weights_array
    
    new_weights = new_values / total_value
    new_weights[~valid_mask] = 0  # Set invalid weights to zero
    
    return new_weights

# =============================================================================
# FILE I/O UTILITY FUNCTIONS
# =============================================================================

def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Parameters:
    -----------
    directory : Union[str, Path]
        Directory path
        
    Returns:
    --------
    Path
        Directory path object
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save object to pickle file.
    
    Parameters:
    -----------
    obj : Any
        Object to save
    filepath : Union[str, Path]
        File path
    """
    filepath = Path(filepath)
    ensure_directory_exists(filepath.parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    
    logger.info(f"Saved object to {filepath}")

def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load object from pickle file.
    
    Parameters:
    -----------
    filepath : Union[str, Path]
        File path
        
    Returns:
    --------
    Any
        Loaded object
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    
    logger.info(f"Loaded object from {filepath}")
    return obj

def save_dataframe(df: pd.DataFrame, 
                  filepath: Union[str, Path],
                  format: str = 'csv',
                  **kwargs) -> None:
    """
    Save DataFrame to file in various formats.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    filepath : Union[str, Path]
        File path
    format : str
        File format ('csv', 'excel', 'parquet', 'json')
    **kwargs
        Additional arguments for pandas save methods
    """
    filepath = Path(filepath)
    ensure_directory_exists(filepath.parent)
    
    if format.lower() == 'csv':
        df.to_csv(filepath, index=False, **kwargs)
    elif format.lower() in ['excel', 'xlsx']:
        df.to_excel(filepath, index=False, **kwargs)
    elif format.lower() == 'parquet':
        df.to_parquet(filepath, index=False, **kwargs)
    elif format.lower() == 'json':
        df.to_json(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved DataFrame to {filepath}")

# =============================================================================
# LOGGING UTILITY FUNCTIONS
# =============================================================================

def setup_logging(log_file: Optional[str] = None,
                 log_level: str = config.LOGGING_LEVEL) -> logging.Logger:
    """
    Set up logging configuration.
    
    Parameters:
    -----------
    log_file : str, optional
        Log file path
    log_level : str
        Logging level
        
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('momentum_replication')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_function_call(func_name: str, **kwargs) -> None:
    """
    Log function call with parameters.
    
    Parameters:
    -----------
    func_name : str
        Function name
    **kwargs
        Function parameters
    """
    param_str = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"Calling {func_name}({param_str})")

# =============================================================================
# VALIDATION UTILITY FUNCTIONS
# =============================================================================

def validate_data_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame has required columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : List[str]
        List of required column names
        
    Returns:
    --------
    bool
        True if all required columns exist
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def validate_date_range(df: pd.DataFrame, 
                       date_column: str,
                       min_date: Optional[datetime] = None,
                       max_date: Optional[datetime] = None) -> bool:
    """
    Validate that DataFrame dates are within expected range.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    date_column : str
        Name of date column
    min_date : datetime, optional
        Minimum expected date
    max_date : datetime, optional
        Maximum expected date
        
    Returns:
    --------
    bool
        True if dates are valid
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found")
    
    dates = pd.to_datetime(df[date_column])
    
    if min_date and dates.min() < pd.Timestamp(min_date):
        raise ValueError(f"Dates start before minimum: {dates.min()} < {min_date}")
    
    if max_date and dates.max() > pd.Timestamp(max_date):
        raise ValueError(f"Dates end after maximum: {dates.max()} > {max_date}")
    
    return True

# =============================================================================
# FORMATTING UTILITY FUNCTIONS
# =============================================================================

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format value as percentage string.
    
    Parameters:
    -----------
    value : float
        Value to format (as decimal, e.g., 0.05 for 5%)
    decimal_places : int
        Number of decimal places
        
    Returns:
    --------
    str
        Formatted percentage string
    """
    if np.isnan(value):
        return "NaN"
    
    return f"{value * 100:.{decimal_places}f}%"

def format_number(value: float, decimal_places: int = 4) -> str:
    """
    Format number with specified decimal places.
    
    Parameters:
    -----------
    value : float
        Value to format
    decimal_places : int
        Number of decimal places
        
    Returns:
    --------
    str
        Formatted number string
    """
    if np.isnan(value):
        return "NaN"
    
    return f"{value:.{decimal_places}f}"

def format_large_number(value: float) -> str:
    """
    Format large numbers with appropriate suffixes (K, M, B).
    
    Parameters:
    -----------
    value : float
        Value to format
        
    Returns:
    --------
    str
        Formatted number string
    """
    if np.isnan(value):
        return "NaN"
    
    if abs(value) >= 1e9:
        return f"{value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.2f}K"
    else:
        return f"{value:.2f}"

# =============================================================================
# CONFIGURATION UTILITY FUNCTIONS
# =============================================================================

def get_jk_combinations() -> List[Tuple[int, int]]:
    """
    Get all (J,K) formation-holding period combinations.
    
    Returns:
    --------
    List[Tuple[int, int]]
        List of (formation_period, holding_period) tuples
    """
    return [(j, k) for j in config.FORMATION_PERIODS for k in config.HOLDING_PERIODS]

def get_results_directory() -> Path:
    """
    Get and ensure results directory exists.
    
    Returns:
    --------
    Path
        Results directory path
    """
    return ensure_directory_exists(config.RESULTS_DIR)

def get_data_directory() -> Path:
    """
    Get and ensure data directory exists.
    
    Returns:
    --------
    Path
        Data directory path
    """
    return ensure_directory_exists(config.DATA_DIR)

def get_figures_directory() -> Path:
    """
    Get and ensure figures directory exists.
    
    Returns:
    --------
    Path
        Figures directory path
    """
    return ensure_directory_exists(config.FIGURES_DIR)