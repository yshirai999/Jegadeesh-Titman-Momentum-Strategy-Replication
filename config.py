"""
Configuration file for Jegadeesh & Titman (1993) momentum strategy replication.
Contains all parameters and settings for the analysis.
"""

import pandas as pd
from datetime import datetime

import config_wrds  # WRDS connection settings

# =============================================================================
# DATA PARAMETERS
# =============================================================================

# Sample period (original paper: 1965-1989)
START_DATE = datetime(1965, 1, 1)
END_DATE = datetime(1989, 12, 31)

# Alternative sample periods for robustness
EXTENDED_START_DATE = datetime(1963, 1, 1)  # Extra data for momentum calculation
EXTENDED_END_DATE = datetime(1989, 12, 31)

# Data source configuration
DATA_SOURCE = "wrds"  # Options: "wrds", "yahoo", "local_csv"
WRDS_USERNAME = config_wrds.WRDS_USERNAME  # Set your WRDS username
WRDS_PASSWORD = config_wrds.WRDS_PASSWORD  # Set your WRDS password

# =============================================================================
# MOMENTUM STRATEGY PARAMETERS
# =============================================================================

# Formation periods (J months) - past return calculation periods
FORMATION_PERIODS = [3, 6, 9, 12]

# Holding periods (K months) - portfolio holding periods
HOLDING_PERIODS = [3, 6, 9, 12]

# Portfolio construction
NUM_PORTFOLIOS = 10  # Number of momentum portfolios (deciles)
PORTFOLIO_WEIGHTING = "equal"  # Options: "equal", "value"

# Skip periods (to avoid microstructure effects)
SKIP_PERIOD = 1  # Skip 1 month between formation and holding

# =============================================================================
# DATA FILTERING PARAMETERS
# =============================================================================

# Minimum price filter (to avoid penny stocks)
MIN_PRICE = 5.0

# Minimum market cap percentile (to avoid micro-cap stocks)
MIN_MARKET_CAP_PERCENTILE = 20

# Minimum number of return observations required
MIN_RETURN_OBSERVATIONS = 24

# Maximum number of consecutive missing returns allowed
MAX_CONSECUTIVE_MISSING = 2

# =============================================================================
# EXCHANGE AND SHARE CODE FILTERS (for CRSP data)
# =============================================================================

# CRSP exchange codes to include (NYSE=1, AMEX=2, NASDAQ=3)
INCLUDED_EXCHANGES = [1, 2, 3]

# CRSP share codes to include (ordinary common shares)
INCLUDED_SHARE_CODES = [10, 11]

# =============================================================================
# OUTPUT PARAMETERS
# =============================================================================

# Results directory
RESULTS_DIR = "results"
DATA_DIR = "data"
FIGURES_DIR = "figures"

# Table formatting
TABLE_FORMAT = "latex"  # Options: "latex", "html", "csv"
DECIMAL_PLACES = 4

# Significance levels for statistical tests
SIGNIFICANCE_LEVELS = [0.01, 0.05, 0.10]

# =============================================================================
# RISK MODEL PARAMETERS
# =============================================================================

# Risk-free rate proxy
RISK_FREE_RATE_SOURCE = "treasury_bills_3m"

# Market portfolio proxy
MARKET_PORTFOLIO = "value_weighted_crsp"  # or "sp500"

# =============================================================================
# REBALANCING PARAMETERS
# =============================================================================

# Rebalancing frequency
REBALANCING_FREQUENCY = "monthly"

# Transaction cost assumptions (basis points)
TRANSACTION_COST_BPS = 50

# =============================================================================
# VALIDATION PARAMETERS
# =============================================================================

# Tolerance levels for validation checks
PRICE_RETURN_TOLERANCE = 0.01  # 1% tolerance for price-return consistency
MARKET_CAP_TOLERANCE = 0.05    # 5% tolerance for market cap calculations

# Data completeness thresholds
MIN_DATA_COMPLETENESS = 0.80  # Minimum 80% data completeness
MIN_STOCKS_PER_MONTH = 100    # Minimum number of stocks per month

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_FILE = "momentum_replication.log"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_formation_holding_combinations():
    """Return all (J,K) combinations of formation and holding periods."""
    combinations = []
    for j in FORMATION_PERIODS:
        for k in HOLDING_PERIODS:
            combinations.append((j, k))
    return combinations

def get_sample_months():
    """Return list of all months in the sample period."""
    return pd.date_range(start=START_DATE, end=END_DATE, freq='M')

def get_extended_sample_months():
    """Return extended date range including pre-sample data for momentum calculation."""
    return pd.date_range(start=EXTENDED_START_DATE, end=END_DATE, freq='M')

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    if START_DATE >= END_DATE:
        errors.append("START_DATE must be before END_DATE")
    
    if not FORMATION_PERIODS or not HOLDING_PERIODS:
        errors.append("FORMATION_PERIODS and HOLDING_PERIODS cannot be empty")
    
    if NUM_PORTFOLIOS < 2:
        errors.append("NUM_PORTFOLIOS must be at least 2")
    
    if MIN_PRICE <= 0:
        errors.append("MIN_PRICE must be positive")
    
    if not (0 <= MIN_MARKET_CAP_PERCENTILE <= 100):
        errors.append("MIN_MARKET_CAP_PERCENTILE must be between 0 and 100")
    
    if errors:
        raise ValueError("Configuration validation errors:\n" + "\n".join(errors))
    
    return True

# Validate configuration on import
validate_config()