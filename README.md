# Jegadeesh & Titman (1993) Momentum Strategy Replication

A Python implementation replicating the momentum strategy from "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency" by Narasimhan Jegadeesh and Sheridan Titman (1993).

## Momentum Strategy Construction and Reported Returns

### Winners and Losers Portfolios formation

- At the end of each month, stocks are ranked by their past J-month returns.
- Stocks are sorted into deciles based on this ranking.
- The winner portfolio consists of the top decile, and the loser portfolio consists of the bottom decile.
- Portfolios are equal-weighted within each leg.

### Zero-cost long–short strategy

- Each month, the strategy goes long the winner portfolio and short the loser portfolio with equal dollar exposure on each side.
- The reported return of a long–short portfolio in a given month is the average return of winners minus the average return of losers.
- Returns are portfolio returns (not cash P&L); scale is implicitly normalized, consistent with the original paper.

### Holding period and overlapping portfolios

- Portfolios are held for K months.
- When K > 1, multiple portfolios formed in different months are held simultaneously.
- The strategy’s monthly return is the simple average of the returns of all active long–short portfolios (overlapping vintages).

### Skip-one-week adjustment

- To match Jegadeesh–Titman’s skip-period convention, the first week of returns is excluded only in the first holding month of each newly formed portfolio.
- Subsequent holding months use full monthly returns.
- The momentum signal itself uses uninterrupted past J month returns (no skip at the signal stage).

### Reported results

The results tables (see below) report:

- Mean monthly return of the long–short (WML) strategy,
- Volatility,
- t-statistics.

These statistics are computed from the time series of monthly strategy returns described above.

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
| J=3 | K=3 | 0.75 | 4.23 | 3.046** | 0.78 | 3.79 | 3.554** |
| J=3 | K=6 | 0.69 | 3.75 | 3.173** | 0.71 | 3.53 | 3.445** |
| J=3 | K=9 | 0.71 | 3.23 | 3.794** | 0.72 | 3.08 | 4.018** |
| J=3 | K=12 | 0.60 | 2.98 | 3.457** | 0.60 | 2.87 | 3.623** |
| J=6 | K=3 | 1.01 | 5.23 | 3.288** | 1.04 | 4.61 | 3.852** |
| J=6 | K=6 | 1.00 | 4.63 | 3.688** | 1.01 | 4.32 | 4.007** |
| J=6 | K=9 | 0.89 | 4.21 | 3.603** | 0.90 | 4.02 | 3.822** |
| J=6 | K=12 | 0.66 | 4.05 | 2.775** | 0.66 | 3.90 | 2.914** |
| J=9 | K=3 | 1.20 | 5.36 | 3.808** | 1.22 | 4.72 | 4.413** |
| J=9 | K=6 | 1.03 | 5.02 | 3.499** | 1.04 | 4.69 | 3.790** |
| J=9 | K=9 | 0.79 | 4.79 | 2.807** | 0.80 | 4.58 | 2.974** |
| J=9 | K=12 | 0.51 | 4.60 | 1.892* | 0.52 | 4.44 | 1.994* |
| J=12 | K=3 | 1.14 | 5.69 | 3.398** | 1.14 | 5.04 | 3.839** |
| J=12 | K=6 | 0.88 | 5.45 | 2.729** | 0.88 | 5.11 | 2.908** |
| J=12 | K=9 | 0.63 | 5.22 | 2.031* | 0.63 | 4.99 | 2.124* |
| J=12 | K=12 | 0.36 | 5.01 | 1.219 | 0.36 | 4.83 | 1.269 |

<!-- END: MOMENTUM_RESULTS_TABLE -->

#### Key Findings

- **All 32 strategies profitable**: WML returns range from .36% to 1.20% monthly on average
- **All highly significant**: Every strategy significant at 5% level (** indicates p > 0.05, and * indicates p > 0.01)
- **Best Results**: J = 9 and K = 3, closely followed by J = 9 and K = 3, very similar to original JT results
- **Skip period benefit**: 1-month skip improves returns, also in line with JT results
- **Robust effect**: Even weakest strategy delivery positive returns

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
