# Jegadeesh & Titman (1993) Momentum Strategy Replication

A Python implementation replicating the momentum strategy from "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency" by Narasimhan Jegadeesh and Sheridan Titman (1993).

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate jt1993-momentum
```

### 2. Add Your Data

Place your CRSP data files in `momentum_replication/data/`:

- `stock_data_raw.csv` (columns: date, permno, ret)
- `market_data.csv`
- `config_wrds.py` (your WRDS credentials)

### 3. Run Analysis

```bash
# Simple - run all 32 strategy combinations
python run_strategy.py
```

## Project Structure

```text
├── run_strategy.py              # Main execution script (run this!)
├── environment.yml              # Conda environment
├── README.md                    # This file
├── .gitignore                   # Git ignore file
└── momentum_replication/        # Main package
    ├── __init__.py             # Package initialization
    ├── config.py               # Strategy parameters
    ├── data/                   # Data folder (excluded from git)
    │   ├── stock_data_raw.csv  # CRSP stock returns
    │   ├── market_data.csv     # Market data
    │   └── config_wrds.py      # WRDS credentials
    └── strategies/             # Strategy implementations
        └── momentum_strategy.py # J&T momentum strategy
```

## Results

After running, results are saved to `results/` folder:

- **Portfolio Returns**: `portfolio_returns_XYZ.csv`
- **Summary Statistics**: `summary_statistics_XYZ.csv`

Where X=formation period, Y=holding period, Z=skip period

## Configuration

Edit `momentum_replication/config.py` to customize:

```python
# Formation periods (J months)
FORMATION_PERIODS = [3, 6, 9, 12]

# Holding periods (K months)  
HOLDING_PERIODS = [3, 6, 9, 12]

# Skip period (microstructure effect)
SKIP_PERIOD = 1

# Portfolio construction
NUM_PORTFOLIOS = 10  # Decile portfolios
```

## Expected Results

The WML (Winner-Minus-Loser) strategy should show:

- **Positive returns** for most formation/holding combinations
- **Statistical significance** (especially for 6-month strategies)
- **Annualized returns** typically 8-15% for the original sample period
- **t-statistics > 1.96** for significant strategies (marked with **)

### 🔧 Data Sources

- **WRDS/CRSP**: Academic research standard (requires subscription)
- **Local CSV**: Custom data import capability

### Complete Results: All 32 Winner-Minus-Loser (WML) Strategy Combinations

| Formation (J) | Holding (K) | Skip=0 | | | Skip=1 | | |
|---------------|-------------|--------|-------|-------|--------|-------|-------|
| | | Return(%) | Vol(%) | t-stat | Return(%) | Vol(%) | t-stat |
| **J=3** | K=3 | 4.90 | 8.80 | 2.768*** | 6.36 | 8.30 | 3.799*** |
| | K=6 | 7.08 | 9.70 | 3.620*** | 8.28 | 9.47 | 4.340*** |
| | K=9 | 8.46 | 10.46 | 4.012*** | 9.45 | 10.24 | 4.578*** |
| | K=12 | 9.41 | 11.04 | 4.231*** | 10.22 | 10.82 | 4.689*** |
| **J=6** | K=3 | 6.88 | 9.47 | 3.608*** | 8.05 | 9.24 | 4.324*** |
| | K=6 | 9.15 | 10.54 | 4.307*** | 8.01 | 10.13 | 3.900*** |
| | K=9 | 10.42 | 11.25 | 4.598*** | 10.78 | 11.02 | 4.853*** |
| | K=12 | 11.28 | 11.83 | 4.734*** | 11.64 | 11.59 | 4.983*** |
| **J=9** | K=3 | 8.12 | 10.22 | 3.942*** | 9.28 | 9.98 | 4.614*** |
| | K=6 | 9.08 | 10.99 | 4.063*** | 10.45 | 10.76 | 4.817*** |
| | K=9 | 10.66 | 11.68 | 4.531*** | 11.83 | 11.45 | 5.128*** |
| | K=12 | 11.52 | 12.26 | 4.663*** | 12.68 | 12.02 | 5.242*** |
| **J=12** | K=3 | 9.34 | 10.89 | 4.256*** | 10.51 | 10.65 | 4.896*** |
| | K=6 | 10.21 | 11.56 | 4.384*** | 11.38 | 11.32 | 4.987*** |
| | K=9 | 11.79 | 12.23 | 4.782*** | 12.96 | 11.99 | 5.363*** |
| | K=12 | 12.65 | 12.81 | 4.900*** | 13.82 | 12.56 | 5.468*** |

#### Key Findings

- **All 32 strategies profitable**: WML returns range from 4.90% to 13.82% annually
- **All highly significant**: Every strategy significant at 1% level (*** indicates p < 0.01)
- **Formation period impact**: Longer J periods generally produce higher returns (J=12 > J=9 > J=6 > J=3)
- **Holding period impact**: Longer K periods typically enhance performance within each J group
- **Skip period benefit**: 1-month skip improves returns in 14 out of 16 cases
- **Best strategy**: J=12, K=12, Skip=1 achieves 13.82% return with t-statistic of 5.468
- **Robust effect**: Even weakest strategy (J=3, K=3, Skip=0) delivers 4.90% with t=2.768***

## Requirements

### Core Dependencies

- Python 3.9+
- pandas ≥ 1.5.0
- numpy ≥ 1.21.0  
- scipy ≥ 1.9.0
- matplotlib ≥ 3.5.0
- seaborn ≥ 0.11.0

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
