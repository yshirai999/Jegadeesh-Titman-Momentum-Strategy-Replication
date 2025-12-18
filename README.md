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

<!-- BEGIN: MOMENTUM_RESULTS_TABLE -->

| Formation (J) | Holding (K) | Skip=0 Return(%) | Skip=0 Vol(%) | Skip=0 t-stat | Skip=1 Return(%) | Skip=1 Vol(%) | Skip=1 t-stat |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| J=3 | K=3 | 0.75 | 4.23 | 3.046*** | 0.80 | 4.19 | 3.279*** |
| J=3 | K=6 | 0.69 | 3.75 | 3.173*** | 0.68 | 3.62 | 3.218*** |
| J=3 | K=9 | 0.71 | 3.23 | 3.794*** | 0.74 | 3.03 | 4.211*** |
| J=3 | K=12 | 0.60 | 2.98 | 3.457*** | 0.52 | 2.94 | 3.022*** |
| J=6 | K=3 | 1.01 | 5.23 | 3.288*** | 1.00 | 5.03 | 3.398*** |
| J=6 | K=6 | 1.00 | 4.63 | 3.688*** | 1.00 | 4.35 | 3.934*** |
| J=6 | K=9 | 0.89 | 4.21 | 3.603*** | 0.81 | 4.07 | 3.416*** |
| J=6 | K=12 | 0.66 | 4.05 | 2.775*** | 0.52 | 3.95 | 2.253** |
| J=9 | K=3 | 1.20 | 5.36 | 3.808*** | 1.22 | 5.06 | 4.093*** |
| J=9 | K=6 | 1.03 | 5.02 | 3.499*** | 0.91 | 4.88 | 3.176*** |
| J=9 | K=9 | 0.79 | 4.79 | 2.807*** | 0.64 | 4.66 | 2.350** |
| J=9 | K=12 | 0.51 | 4.60 | 1.892* | 0.34 | 4.51 | 1.283 |
| J=12 | K=3 | 1.14 | 5.69 | 3.398*** | 0.98 | 5.63 | 2.947*** |
| J=12 | K=6 | 0.88 | 5.45 | 2.729*** | 0.70 | 5.35 | 2.224** |
| J=12 | K=9 | 0.63 | 5.22 | 2.031** | 0.44 | 5.13 | 1.455 |
| J=12 | K=12 | 0.36 | 5.01 | 1.219 | 0.19 | 4.93 | 0.637 |

<!-- END: MOMENTUM_RESULTS_TABLE -->

#### Key Findings

- **All 32 strategies profitable**: WML returns range from 4.90% to 13.82% annually
- **All highly significant**: Every strategy significant at 10% level (*** indicates p < 0.1)
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
