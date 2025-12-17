# Jegadeesh & Titman (1993) Momentum Strategy Replication

A comprehensive Python implementation replicating the seminal momentum strategy from "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency" by Narasimhan Jegadeesh and Sheridan Titman (1993).

## Overview

This project provides a complete, academically rigorous replication of the J&T momentum strategy with:

- **Multiple formation periods**: 3, 6, 9, and 12 months
- **Multiple holding periods**: 3, 6, 9, and 12 months  
- **Overlapping portfolio construction** as per the original methodology
- **Statistical testing** with t-statistics and significance levels
- **Risk-adjusted performance** using CAPM regressions
- **Publication-ready tables** matching the original paper format

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate jt1993-momentum

# Verify installation
conda list pandas numpy matplotlib
```

### 2. Configuration

Edit `config.py` to customize:

- Sample period dates
- Data source (WRDS, Yahoo Finance, or local CSV)
- Formation and holding periods
- Portfolio construction parameters

### 3. Run Analysis

```bash
python main.py
```

## Project Structure

### Core Modules

```text
├── config.py                 # Configuration parameters
├── main.py                   # Main execution script
├── environment.yml           # Conda environment specification
└── README.md                # This file
```

### Data Management

```text
├── data_loader.py           # Multi-source data loading
├── data_cleaner.py          # Academic-standard data preprocessing  
├── data_validator.py        # Comprehensive data quality validation
└── date_utils.py            # Specialized date handling functions
```

### Strategy Implementation

```text
├── momentum_calculator.py   # Past return calculations
├── portfolio_builder.py     # Winner/loser portfolio formation
├── rebalancer.py           # Overlapping portfolio management
├── momentum_strategy.py     # Main strategy orchestration
└── returns_calculator.py    # Performance calculations
```

### Analysis & Results

```text
├── statistical_tests.py    # t-tests and significance testing
├── performance_metrics.py   # Risk and return metrics
├── risk_analysis.py        # Risk-adjusted performance
├── results_generator.py    # Publication tables
├── plotting.py             # Comprehensive visualizations
├── latex_tables.py         # LaTeX formatted tables
└── utils.py                # General utility functions
```

## Key Features

### 📊 Academic Methodology

- **Exact J&T Replication**: Formation/holding periods (3,6,9,12 months)
- **Survivorship Bias Handling**: Point-in-time data usage
- **Skip Period Implementation**: 1-month gap between formation and holding
- **Overlapping Portfolios**: Proper monthly rebalancing

### 📈 Statistical Analysis

- **Significance Testing**: t-statistics at 1%, 5%, and 10% levels
- **CAPM Regressions**: Risk-adjusted alphas and betas
- **Performance Metrics**: Sharpe ratios, maximum drawdown, volatility
- **Winner-Loser Spreads**: Long-short portfolio returns

### 🔧 Data Sources

- **WRDS/CRSP**: Academic research standard (requires subscription)
- **Local CSV**: Custom data import capability

### 📋 Output

- **Table I Format**: Monthly returns matching original paper
- **LaTeX Tables**: Academic publication formatting
- **Statistical Significance**: Proper significance markers (*, **, ***)
- **Performance Summary**: Comprehensive results tables

## Sample Output

The analysis generates tables like the original J&T paper:

```text
Table I: Average Monthly Returns of Momentum Portfolios
Formation Period (J months) | 3      | 6      | 9      | 12
Winner Portfolio (P10)      | 1.68***| 1.43** | 1.79***| 1.95***
Loser Portfolio (P1)        |-0.49** |-0.85***|-0.36*  |-0.24
Winner-Loser Spread         | 2.17***| 2.28***| 2.15***| 2.19***

* p < 0.10, ** p < 0.05, *** p < 0.01
```

## Configuration Options

### Data Settings

```python
# In config.py
DATA_SOURCE = 'wrds'  # 'wrds' or 'csv'
START_DATE = '1965-01-01'
END_DATE = '1989-12-31'
```

### Strategy Parameters

```python
FORMATION_PERIODS = [3, 6, 9, 12]    # Formation period lengths
HOLDING_PERIODS = [3, 6, 9, 12]      # Holding period lengths  
NUM_PORTFOLIOS = 10                   # Number of momentum deciles
SKIP_PERIOD = 1                       # Months between formation and holding
```

### Portfolio Construction

```python
WEIGHTING_SCHEME = 'equal'           # 'equal' or 'value'
MIN_PRICE = 5.0                      # Minimum stock price filter
MIN_MARKET_CAP_PERCENTILE = 0.2      # NYSE market cap filter
```

## Results Directory Structure

After running the analysis:

```text
├── results/
│   ├── momentum_returns_summary.csv
│   ├── performance_metrics.csv
│   ├── statistical_significance.csv
│   └── winner_loser_spreads.csv
├── figures/
│   ├── cumulative_returns.png
│   ├── winner_loser_heatmap.png
│   └── portfolio_performance.png
└── tables/
    ├── table_i_monthly_returns.tex
    ├── performance_metrics.tex
    └── complete_analysis.tex
```

## Requirements

### Core Dependencies

- Python 3.9+
- pandas ≥ 1.5.0
- numpy ≥ 1.21.0  
- scipy ≥ 1.9.0
- matplotlib ≥ 3.5.0
- seaborn ≥ 0.11.0

### Data Sources

- **WRDS Access**: Academic subscription required for CRSP data
- **Local Data**: CSV format with columns: date, ticker, price, return, market_cap

## Troubleshooting

### Common Issues

1. **WRDS Connection**: Ensure WRDS credentials are configured
2. **Data Coverage**: Check sample period matches data availability
3. **Memory Usage**: Large datasets may require chunked processing
4. **Missing Data**: Validation module reports data quality issues

### Performance Tips

- Use `DATA_SAMPLE_SIZE` in config.py for testing with subset of data
- Enable `ENABLE_CACHING` for faster repeated runs
- Set `PARALLEL_PROCESSING = True` for multi-core execution

## Citation

This work is based on:

```bibtex
@article{jegadeesh1993returns,
  title={Returns to buying winners and selling losers: Implications for stock market efficiency},
  author={Jegadeesh, Narasimhan and Titman, Sheridan},
  journal={The Journal of Finance},
  volume={48},
  number={1},
  pages={65--91},
  year={1993},
  publisher={Wiley Online Library}
}
```

## License

This project is for academic and educational purposes. Please ensure compliance with data provider terms of service when using WRDS or other commercial data sources.

## Support

For questions or issues:

1. Check configuration settings in `config.py`
2. Review data validation output from `data_validator.py`
3. Enable debug logging: `LOGGING_LEVEL = 'DEBUG'` in config.py
4. Examine intermediate results in `/results` directory
