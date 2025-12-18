"""
Configuration file for Jegadeesh & Titman (1993) momentum strategy replication.
Contains all parameters and settings for the analysis.
"""

import pandas as pd
from datetime import datetime
import sys
import os

# Import WRDS config from data folder (now in same directory)
from data import config_wrds  # WRDS connection settings

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
DATA_SOURCE = "local_csv"  # Options: "wrds", "local_csv"
WRDS_USERNAME = config_wrds.WRDS_USERNAME  # Set your WRDS username
WRDS_PASSWORD = config_wrds.WRDS_PASSWORD  # Set your WRDS password (may not be used with Duo)

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

# Table formatting
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

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_FILE = "momentum_replication.log"