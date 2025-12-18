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
from typing import Tuple, Dict, List

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load stock and market data from CSV files.
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Stock data and market data
    """
    # Get data directory path (go up one level from momentum strategy folder)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    # Load stock data
    stock_file = os.path.join(data_dir, 'stock_data_raw.csv')
    if not os.path.exists(stock_file):
        raise FileNotFoundError(f"Stock data file not found: {stock_file}")
    
    stock_data = pd.read_csv(stock_file)
    
    # Load market data
    market_file = os.path.join(data_dir, 'market_data.csv')
    if not os.path.exists(market_file):
        raise FileNotFoundError(f"Market data file not found: {market_file}")
    
    market_data = pd.read_csv(market_file)
    
    print(f"✅ Loaded stock data: {stock_data.shape}")
    print(f"✅ Loaded market data: {market_data.shape}")
    
    return stock_data, market_data

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

def calculate_momentum_scores(returns_matrix: pd.DataFrame, 
                            formation_period: int = 6,
                            skip_period: int = 1) -> pd.DataFrame:
    """
    Calculate momentum scores for each stock at each date.
    
    Parameters:
    -----------
    returns_matrix : pd.DataFrame
        Monthly returns with dates as index, stocks as columns
    formation_period : int
        Number of months to look back for momentum calculation
    skip_period : int
        Number of months to skip to avoid microstructure effects
        
    Returns:
    --------
    pd.DataFrame
        Momentum scores (cumulative returns over formation period)
    """
    print(f"📊 Calculating momentum scores (J={formation_period}, skip={skip_period})")
    
    # Calculate cumulative returns over formation period
    # We need to skip the most recent month to avoid microstructure effects
    momentum_scores = pd.DataFrame(index=returns_matrix.index, columns=returns_matrix.columns)
    
    for i in range(formation_period + skip_period, len(returns_matrix)):
        current_date = returns_matrix.index[i]
        
        # Get returns for formation period (skipping the most recent skip_period months)
        start_idx = i - formation_period - skip_period
        end_idx = i - skip_period
        
        formation_returns = returns_matrix.iloc[start_idx:end_idx]
        
        # Calculate cumulative return: (1+r1)*(1+r2)*...*(1+rJ) - 1
        momentum_scores.loc[current_date] = (1 + formation_returns).prod() - 1
    
    print(f"✅ Momentum scores calculated for {momentum_scores.count().sum()} stock-month observations")
    
    return momentum_scores

def create_momentum_portfolios(returns_matrix: pd.DataFrame, 
                             momentum_scores: pd.DataFrame,
                             n_portfolios: int = 10) -> pd.DataFrame:
    """
    Create winner and loser portfolios for momentum strategy.
    
    Parameters:
    -----------
    returns_matrix : pd.DataFrame
        Monthly returns matrix
    momentum_scores : pd.DataFrame
        Momentum scores for portfolio formation
        
    Returns:
    --------
    pd.DataFrame
        Portfolio returns for Winners (P10), Losers (P1), and WML strategy
    """
    print("📈 Creating winner and loser momentum portfolios")
    
    # Initialize return series
    winners_returns = pd.Series(index=returns_matrix.index, dtype=float, name='Winners')
    losers_returns = pd.Series(index=returns_matrix.index, dtype=float, name='Losers')
    
    # For each formation date, create winner/loser portfolios
    for formation_date in momentum_scores.dropna().index:
        # Get momentum scores for this date
        current_scores = momentum_scores.loc[formation_date].dropna()
            
        # Calculate portfolio cutoffs
        portfolio_size = len(current_scores) // n_portfolios
        
        # Sort stocks by momentum score
        sorted_stocks = current_scores.sort_values()
        
        # Bottom portfolio (Losers - P1)
        losers = sorted_stocks.iloc[:portfolio_size].index
        
        # Top portfolio (Winners - Pn)  
        winners = sorted_stocks.iloc[-portfolio_size:].index
        
        # Calculate next month returns
        next_month_idx = returns_matrix.index.get_loc(formation_date) + 1
        
        if next_month_idx < len(returns_matrix):
            next_month_date = returns_matrix.index[next_month_idx]
            
            # Winner portfolio return
            winner_rets = returns_matrix.loc[next_month_date, winners]
            winners_returns[next_month_date] = winner_rets.mean()
            
            # Loser portfolio return
            loser_rets = returns_matrix.loc[next_month_date, losers]
            losers_returns[next_month_date] = loser_rets.mean()
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Losers': losers_returns,
        'Winners': winners_returns
    })
    
    # Add Winner-Loser strategy
    results['WML'] = results['Winners'] - results['Losers']
    
    print(f"✅ Portfolio returns calculated: {results.shape}")
    
    return results

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
        mean_ret = ret_series.mean() * 12  # Annualized
        volatility = ret_series.std() * np.sqrt(12)  # Annualized
        sharpe = mean_ret / volatility if volatility > 0 else np.nan
        
        # Calculate t-statistic for mean return
        t_stat = ret_series.mean() / (ret_series.std() / np.sqrt(len(ret_series))) if ret_series.std() > 0 else np.nan
        
        # Add significance asterisks
        t_stat_str = f"{t_stat:.3f}" if not np.isnan(t_stat) else "N/A"
        if not np.isnan(t_stat):
            if abs(t_stat) > 2.576:  # 1% significance
                t_stat_str += "***"
            elif abs(t_stat) > 1.96:  # 5% significance
                t_stat_str += "**"
            elif abs(t_stat) > 1.645:  # 10% significance
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
                         skip_period: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete momentum strategy analysis.
    
    Parameters:
    -----------
    formation_period : int
        Months to look back for momentum calculation (J)
    holding_period : int
        Months to hold portfolio (K) - not implemented yet, using 1 month
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
    print(f"Skip Period: {skip_period} month(s)")
    print()
    
    # Step 1: Load data
    print("STEP 1: Loading data...")
    stock_data, market_data = load_data()
    
    # Step 2: Prepare returns matrix
    print("\nSTEP 2: Preparing returns data...")
    returns_matrix = prepare_returns_data(stock_data)
    
    # Step 3: Calculate momentum scores
    print(f"\nSTEP 3: Calculating momentum scores...")
    momentum_scores = calculate_momentum_scores(returns_matrix, formation_period, skip_period)
    
    # Step 4: Create momentum portfolios
    print(f"\nSTEP 4: Creating momentum portfolios...")
    portfolio_returns = create_momentum_portfolios(returns_matrix, momentum_scores, n_portfolios)
    
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
        print(f"Skip periods: [0, {config.SKIP_PERIOD}]")
        print(f"Total strategies: {len(config.FORMATION_PERIODS) * len(config.HOLDING_PERIODS) * 2}")
        print(f"Results will be saved to: {results_dir}")
        print("=" * 80)
        
        strategy_count = 0
        
        # Run all combinations
        for formation in config.FORMATION_PERIODS:
            for holding in config.HOLDING_PERIODS:
                for skip in [0, config.SKIP_PERIOD]:
                    strategy_count += 1
                    
                    print(f"\n[{strategy_count}/32] Running strategy J={formation}, K={holding}, Skip={skip}")
                    
                    try:
                        # Run strategy
                        returns, stats = run_momentum_strategy(
                            formation_period=formation,
                            holding_period=holding,
                            n_portfolios=config.NUM_PORTFOLIOS,
                            skip_period=skip
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