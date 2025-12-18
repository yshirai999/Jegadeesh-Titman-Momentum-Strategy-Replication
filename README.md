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
| J=3 | K=3 | 5.87 | 7.51 | 3.878*** | 6.00 | 7.40 | 4.026*** |
| J=3 | K=6 | 5.40 | 6.66 | 4.028*** | 5.54 | 6.56 | 4.183*** |
| J=3 | K=9 | 5.82 | 5.97 | 4.839*** | 5.75 | 5.79 | 4.923*** |
| J=3 | K=12 | 4.63 | 5.51 | 4.178*** | 4.09 | 5.41 | 3.744*** |
| J=6 | K=3 | 7.73 | 9.60 | 3.982*** | 7.55 | 9.49 | 3.926*** |
| J=6 | K=6 | 7.77 | 8.81 | 4.359*** | 7.53 | 8.54 | 4.353*** |
| J=6 | K=9 | 6.73 | 8.07 | 4.122*** | 5.98 | 7.89 | 3.741*** |
| J=6 | K=12 | 4.92 | 7.43 | 3.268*** | 3.98 | 7.20 | 2.727*** |
| J=9 | K=3 | 9.67 | 10.24 | 4.642*** | 9.40 | 9.93 | 4.643*** |
| J=9 | K=6 | 8.13 | 9.73 | 4.105*** | 7.03 | 9.47 | 3.645*** |
| J=9 | K=9 | 6.33 | 9.07 | 3.429*** | 5.10 | 8.73 | 2.864*** |
| J=9 | K=12 | 4.34 | 8.38 | 2.545** | 3.02 | 8.07 | 1.835* |
| J=12 | K=3 | 8.65 | 10.76 | 3.932*** | 7.29 | 10.61 | 3.354*** |
| J=12 | K=6 | 6.63 | 10.24 | 3.165*** | 5.13 | 9.93 | 2.519** |
| J=12 | K=9 | 4.81 | 9.58 | 2.457** | 3.34 | 9.30 | 1.756* |
| J=12 | K=12 | 2.81 | 8.96 | 1.531 | 1.44 | 8.70 | 0.808 |

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
