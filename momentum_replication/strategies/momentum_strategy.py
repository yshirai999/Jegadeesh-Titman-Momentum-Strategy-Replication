"""
Jegadeesh & Titman (1993) Momentum Strategy Replication
=============================================================

This script implements the core momentum strategy from Jegadeesh & Titman (1993).

Key steps:
1. Load stock return data
2. Calculate momentum scores (past J-month returns)
3. Sort stocks into portfolios (winners/losers)
4. Calculate portfolio returns for holding period K
5. Generate results table
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from typing import Tuple, Dict, List, Optional
from collections import defaultdict

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

from typing import Tuple
import os
import pandas as pd

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load stock, market, and first-week return data from CSV files.
    Returns: (stock_data, market_data, first_week_returns)
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    stock_file = os.path.join(data_dir, 'stock_data_raw.csv')
    market_file = os.path.join(data_dir, 'market_data.csv')
    week_file  = os.path.join(data_dir, 'first_week_returns.csv')

    if not os.path.exists(stock_file):
        raise FileNotFoundError(f"Stock data file not found: {stock_file}")
    if not os.path.exists(market_file):
        raise FileNotFoundError(f"Market data file not found: {market_file}")
    if not os.path.exists(week_file):
        raise FileNotFoundError(f"First-week returns file not found: {week_file}")

    stock_data = pd.read_csv(stock_file)
    market_data = pd.read_csv(market_file)
    first_week = pd.read_csv(week_file)

    print(f"Loaded stock data: {stock_data.shape}")
    print(f"Loaded market data: {market_data.shape}")
    print(f"Loaded first-week returns: {first_week.shape}")

    return stock_data, market_data, first_week

def prepare_returns_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare monthly return data in matrix format.
    
    Parameters:
    -----------
    stock_data : pd.DataFrame
        Raw stock data with returns (columns: date, permno, ret)
        
    Returns:
    --------
    pd.DataFrame
        Returns matrix with dates as rows, permno as columns and monthly returns as values
    """
    # Convert date column
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    
    # Convert returns to numeric, handling missing values
    stock_data['ret'] = pd.to_numeric(stock_data['ret'], errors='coerce')
    
    # Remove invalid returns (missing, extreme values)
    stock_data = stock_data.dropna(subset=['ret'])
    stock_data = stock_data[(stock_data['ret'] > -1) & (stock_data['ret'] < 10)]  # Remove extreme values
    
    # Create returns matrix: dates as rows, stocks as columns
    returns_matrix = stock_data.pivot_table(
        index='date', 
        columns='permno', 
        values='ret', 
        aggfunc='first'
    )
    
    # Sort by date
    returns_matrix = returns_matrix.sort_index()
    
    print(f"✅ Returns matrix shape: {returns_matrix.shape}")
    print(f"📅 Date range: {returns_matrix.index.min()} to {returns_matrix.index.max()}")
    
    return returns_matrix

def calculate_momentum_scores(
        returns_matrix: pd.DataFrame,
        formation_period: int = 6
    ) -> pd.DataFrame:
    """
    Momentum score at date t = product_{m=t-J,...,t-1}(1+r_m) - 1
    """
    momentum_scores = pd.DataFrame(index=returns_matrix.index, columns=returns_matrix.columns)
    for i in range(formation_period, len(returns_matrix.index)):
        current_date = returns_matrix.index[i]

        start_idx = i - formation_period
        end_idx = i  # up to t-1

        formation_returns = returns_matrix.iloc[start_idx:end_idx]
        momentum_scores.loc[current_date] = (1 + formation_returns).prod(axis=0, skipna=False) - 1
    print(f"✅ Momentum scores calculated for {momentum_scores.count().sum()} stock-month observations")
    return momentum_scores

def create_momentum_portfolios(
    returns_matrix: pd.DataFrame,
    momentum_scores: pd.DataFrame,
    stock_data: pd.DataFrame,
    holding_period: int = 6,
    n_portfolios: int = 10,
    value_weighted: bool = True,
    skip_week: bool = False,
    first_week_returns: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:

    returns_matrix = returns_matrix.copy()
    returns_matrix.index = pd.to_datetime(returns_matrix.index)
    momentum_scores = momentum_scores.copy()
    momentum_scores.index = pd.to_datetime(momentum_scores.index)

    stock_data = stock_data.copy()
    stock_data['date'] = pd.to_datetime(stock_data['date'])

    mcap_matrix = stock_data.pivot_table(
        index='date',
        columns='permno',
        values='market_cap',
        aggfunc='first'
    ).reindex(index=returns_matrix.index)

    first_week_matrix = None
    if first_week_returns is not None:
        fw = first_week_returns.copy()
        fw['month_start'] = pd.to_datetime(fw['month_start'])
        # pivot: index=month_start, columns=permno, values=first_week_ret
        first_week_matrix = fw.pivot(index='month_start', columns='permno', values='first_week_ret')

    # Store cohort-level WML returns
    wml_vintages = defaultdict(list)

    formation_dates = momentum_scores.dropna(how='all').index

    for formation_date in formation_dates:
        if formation_date not in returns_matrix.index:
            continue

        scores = momentum_scores.loc[formation_date].dropna()
        if len(scores) < n_portfolios * 10:
            continue

        sorted_permnos = scores.sort_values()
        portfolio_size = len(sorted_permnos) // n_portfolios
        if portfolio_size < 1:
            continue

        losers = sorted_permnos.iloc[:portfolio_size].index
        winners = sorted_permnos.iloc[-portfolio_size:].index

        # Formation weights
        if value_weighted:
            w_w = mcap_matrix.loc[formation_date, winners]
            w_l = mcap_matrix.loc[formation_date, losers]

            if w_w.isna().all() or w_w.fillna(0).sum() <= 0:
                w_w = pd.Series(1.0, index=winners)
            if w_l.isna().all() or w_l.fillna(0).sum() <= 0:
                w_l = pd.Series(1.0, index=losers)

            w_w = w_w.fillna(0)
            w_l = w_l.fillna(0)

            w_w = w_w / w_w.sum()
            w_l = w_l / w_l.sum()
        else:
            w_w = pd.Series(1.0 / len(winners), index=winners)
            w_l = pd.Series(1.0 / len(losers), index=losers)

        start_pos = returns_matrix.index.get_loc(formation_date)

        for k in range(1, holding_period + 1):
            hold_pos = start_pos + k
            if hold_pos >= len(returns_matrix.index):
                break

            hold_date = returns_matrix.index[hold_pos]

            r_w = returns_matrix.loc[hold_date, winners]
            r_l = returns_matrix.loc[hold_date, losers]

            if skip_week and (k == 1) and (first_week_matrix is not None):
                month_start = hold_date.to_period('M').to_timestamp(how='start')

                if month_start in first_week_matrix.index:
                    fw_w = first_week_matrix.loc[month_start, winners]
                    fw_l = first_week_matrix.loc[month_start, losers]

                    # If first-week ret is missing -> keep monthly.
                    # If first-week ret <= -1 -> undefined ex-week -> treat as missing.
                    fw_w = fw_w.where(fw_w > -1, np.nan)
                    fw_l = fw_l.where(fw_l > -1, np.nan)

                    # R_ex1wk = (1+R_month)/(1+R_1wk) - 1
                    r_w = pd.Series(
                        np.where(fw_w.notna(), (1.0 + r_w) / (1.0 + fw_w) - 1.0, r_w),
                        index=winners
                    )
                    r_l = pd.Series(
                        np.where(fw_l.notna(), (1.0 + r_l) / (1.0 + fw_l) - 1.0, r_l),
                        index=losers
                    )

            mask_w = r_w.notna()
            mask_l = r_l.notna()

            if mask_w.sum() == 0 or mask_l.sum() == 0:
                continue

            ww_eff = w_w[mask_w]
            ww_eff = ww_eff / ww_eff.sum()

            wl_eff = w_l[mask_l]
            wl_eff = wl_eff / wl_eff.sum()

            r_w_cohort = (ww_eff * r_w[mask_w]).sum()
            r_l_cohort = (wl_eff * r_l[mask_l]).sum()

            wml_vintages[hold_date].append(float(r_w_cohort - r_l_cohort))

    # Aggregate overlapping cohorts
    idx = returns_matrix.index #dates of returns matrix
    wml_series = pd.Series(index=idx, dtype=float, name='WML')

    for d in idx:
        if d in wml_vintages and len(wml_vintages[d]) > 0:
            wml_series.loc[d] = np.mean(wml_vintages[d])

    out = pd.DataFrame({'WML': wml_series})
    return out

def calculate_summary_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics for the momentum strategy.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Monthly strategy returns
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics table
    """
    print("📊 Calculating summary statistics")
    
    stats = []
    
    for col in returns.columns:
        ret_series = returns[col].dropna()
        
        # Calculate statistics
        mean_ret = ret_series.mean()
        volatility = ret_series.std()
        sharpe = mean_ret / volatility if volatility > 0 else np.nan
        
        # Calculate t-statistic for mean return
        t_stat = ret_series.mean() / (ret_series.std() / np.sqrt(len(ret_series))) if ret_series.std() > 0 else np.nan
        
        # Add significance asterisks
        t_stat_str = f"{t_stat:.3f}" if not np.isnan(t_stat) else "N/A"
        if not np.isnan(t_stat):
            if abs(t_stat) > 6.31:  # 10% significance
                t_stat_str += "***"
            elif abs(t_stat) > 2.23:  # 5% significance
                t_stat_str += "**"
            elif abs(t_stat) > 1.81:  # 1% significance
                t_stat_str += "*"
        
        stats.append({
            'Portfolio': col,
            'Mean Return (%)': mean_ret * 100,
            'Volatility (%)': volatility * 100,
            'Sharpe Ratio': sharpe,
            't-statistic': t_stat_str,
            'Observations': len(ret_series)
        })
    
    stats_df = pd.DataFrame(stats)
    
    return stats_df

def run_momentum_strategy(formation_period: int = 6, 
                         holding_period: int = 6,
                         n_portfolios: int = 10,
                         skip_week: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete momentum strategy analysis.
    
    Parameters:
    -----------
    formation_period : int
        Months to look back for momentum calculation (J)
    holding_period : int
        Months to hold portfolio (K)
    n_portfolios : int
        Number of momentum portfolios
    skip_period : int
        Months to skip between formation and holding
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Portfolio returns and summary statistics
    """
    print("=" * 60)
    print("JEGADEESH & TITMAN (1993) MOMENTUM STRATEGY")
    print("=" * 60)
    print(f"Formation Period (J): {formation_period} months")
    print(f"Holding Period (K): {holding_period} months")
    print(f"Number of Portfolios: {n_portfolios}")
    print(f"Skip Week: {skip_week}")
    print()
    
    # Step 1: Load data
    print("STEP 1: Loading data...")
    stock_data, market_data, first_week = load_data()
    
    # Step 2: Prepare returns matrix
    print("\nSTEP 2: Preparing returns data...")
    returns_matrix = prepare_returns_data(stock_data)
    
    # Step 3: Calculate momentum scores
    print(f"\nSTEP 3: Calculating momentum scores...")
    momentum_scores = calculate_momentum_scores(
        returns_matrix,
        formation_period=formation_period)
    
    # Step 4: Create momentum portfolios
    print(f"\nSTEP 4: Creating momentum portfolios...")
    portfolio_returns = create_momentum_portfolios(
        returns_matrix,
        momentum_scores,
        stock_data,
        holding_period=holding_period,
        n_portfolios=n_portfolios,
        value_weighted=False,
        skip_week=skip_week,                 # JT-style skip approximation
        first_week_returns=first_week)

    # Step 5: Summary statistics
    print(f"\nSTEP 5: Computing summary statistics...")
    summary_stats = calculate_summary_statistics(portfolio_returns)
    
    return portfolio_returns, summary_stats

def main():
    """
    Main execution function - runs all strategy combinations from config.
    """
    try:
        # Create results directory
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        print("=" * 80)
        print("JEGADEESH & TITMAN (1993) MOMENTUM STRATEGY - ALL COMBINATIONS")
        print("=" * 80)
        print(f"Formation periods: {config.FORMATION_PERIODS}")
        print(f"Holding periods: {config.HOLDING_PERIODS}")
        print(f"Skip week: [{config.SKIP_WEEK}]")
        print(f"Total strategies: {len(config.FORMATION_PERIODS) * len(config.HOLDING_PERIODS) * 2}")
        print(f"Results will be saved to: {results_dir}")
        print("=" * 80)
        
        strategy_count = 0
        
        # Run all combinations
        for formation in config.FORMATION_PERIODS:
            for holding in config.HOLDING_PERIODS:
                for skip in config.SKIP_WEEK:
                    strategy_count += 1
                    
                    print(f"\n[{strategy_count}/32] Running strategy J={formation}, K={holding}, Skip={skip}")
                    
                    try:
                        # Run strategy
                        returns, stats = run_momentum_strategy(
                            formation_period=formation,
                            holding_period=holding,
                            n_portfolios=config.NUM_PORTFOLIOS,
                            skip_week=skip
                        )
                        
                        # Create file names
                        file_suffix = f"{formation}{holding}{skip}"
                        returns_file = f"portfolio_returns_{file_suffix}.csv"
                        stats_file = f"summary_statistics_{file_suffix}.csv"
                        
                        # Save results
                        returns.to_csv(os.path.join(results_dir, returns_file))
                        stats.to_csv(os.path.join(results_dir, stats_file), index=False)
                        
                        # Display key result
                        if 'WML' in stats['Portfolio'].values:
                            wml_stats = stats[stats['Portfolio'] == 'WML']
                            wml_return = wml_stats['Mean Return (%)'].iloc[0]
                            wml_tstat = wml_stats['t-statistic'].iloc[0]
                            print(f"   WML Return: {wml_return:.2f}% | t-stat: {wml_tstat}")
                        
                        print(f"   ✅ Saved: {returns_file}, {stats_file}")
                        
                    except Exception as e:
                        print(f"   ❌ Error in strategy J={formation}, K={holding}, Skip={skip}: {e}")
                        continue
        
        print(f"\n" + "=" * 60)
        print(f"🎯 COMPLETED: {strategy_count} strategies processed")
        print(f"📁 Results saved to: {results_dir}")
        print("   Files: portfolio_returns_XYZ.csv and summary_statistics_XYZ.csv")
        print("   Where X=formation, Y=holding, Z=skip period")
        print("=" * 60)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()