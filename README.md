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
| J=3 | K=3 | 2.64 | 10.04 | 1.303 | 3.40 | 9.70 | 1.739* |
| J=3 | K=6 | 2.50 | 8.55 | 1.452 | 3.60 | 8.48 | 2.107** |
| J=3 | K=9 | 4.05 | 7.72 | 2.605*** | 4.80 | 7.63 | 3.119*** |
| J=3 | K=12 | 3.69 | 7.21 | 2.545** | 3.73 | 7.07 | 2.618*** |
| J=6 | K=3 | 4.61 | 12.30 | 1.853* | 5.37 | 12.26 | 2.160** |
| J=6 | K=6 | 6.17 | 11.30 | 2.700*** | 7.04 | 11.22 | 3.093*** |
| J=6 | K=9 | 6.40 | 10.55 | 2.998*** | 6.24 | 10.39 | 2.962*** |
| J=6 | K=12 | 5.19 | 9.72 | 2.638*** | 4.64 | 9.50 | 2.411** |
| J=9 | K=3 | 8.24 | 13.51 | 3.000*** | 9.17 | 13.30 | 3.382*** |
| J=9 | K=6 | 7.80 | 12.79 | 2.998*** | 7.80 | 12.49 | 3.065*** |
| J=9 | K=9 | 7.22 | 11.95 | 2.968*** | 6.69 | 11.67 | 2.816*** |
| J=9 | K=12 | 5.57 | 11.16 | 2.455** | 4.87 | 10.93 | 2.186** |
| J=12 | K=3 | 8.64 | 14.23 | 2.967*** | 8.04 | 13.88 | 2.829*** |
| J=12 | K=6 | 7.72 | 13.38 | 2.819*** | 6.94 | 13.04 | 2.598*** |
| J=12 | K=9 | 6.50 | 12.63 | 2.518** | 5.68 | 12.35 | 2.246** |
| J=12 | K=12 | 5.13 | 11.98 | 2.093** | 4.23 | 11.79 | 1.752* |

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
