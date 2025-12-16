"""
Rebalancer module for Jegadeesh & Titman (1993) momentum strategy replication.
Handles portfolio rebalancing logic and overlapping portfolio management.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import warnings

import config
from date_utils import DateUtils
from utils import calculate_portfolio_return

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOGGING_LEVEL))
logger = logging.getLogger(__name__)

class PortfolioRebalancer:
    """
    Class for managing portfolio rebalancing in momentum strategies.
    Handles overlapping portfolios, weight adjustments, and turnover calculation.
    """
    
    def __init__(self):
        """Initialize PortfolioRebalancer."""
        self.rebalancing_history = {}
        self.turnover_statistics = {}
    
    def create_overlapping_strategy(self, 
                                   portfolios: pd.DataFrame,
                                   stock_data: pd.DataFrame,
                                   j: int, k: int) -> pd.DataFrame:
        """
        Create overlapping portfolio strategy following JT methodology.
        
        In the JT approach, a new portfolio is formed every month but held for K months.
        This creates K overlapping portfolios at any given time.
        
        Parameters:
        -----------
        portfolios : pd.DataFrame
            Portfolio compositions from PortfolioBuilder
        stock_data : pd.DataFrame
            Stock return data
        j : int
            Formation period (months)
        k : int
            Holding period (months)
            
        Returns:
        --------
        pd.DataFrame
            Overlapping portfolio holdings by date
        """
        logger.info(f"Creating overlapping strategy for ({j},{k}) portfolios...")
        
        stock_id = self._get_stock_id_column(stock_data)
        
        # Get all dates where we have portfolio holdings
        holding_dates = self._generate_holding_dates(portfolios, stock_data)
        
        overlapping_holdings = []
        
        for date in holding_dates:
            # Find all portfolios active on this date
            active_portfolios = self._get_active_portfolios(portfolios, date)
            
            if active_portfolios.empty:
                continue
            
            # Aggregate overlapping portfolios
            monthly_holdings = self._aggregate_overlapping_portfolios(
                active_portfolios, date, k, stock_id
            )
            
            overlapping_holdings.append(monthly_holdings)
        
        if overlapping_holdings:
            result = pd.concat(overlapping_holdings, ignore_index=True)
            logger.info(f"Created overlapping strategy with {len(result)} portfolio-date observations")
            return result
        else:
            return pd.DataFrame()
    
    def _generate_holding_dates(self, 
                               portfolios: pd.DataFrame,
                               stock_data: pd.DataFrame) -> List[pd.Timestamp]:
        """
        Generate list of dates for portfolio holdings.
        
        Parameters:
        -----------
        portfolios : pd.DataFrame
            Portfolio compositions
        stock_data : pd.DataFrame
            Stock return data
            
        Returns:
        --------
        List[pd.Timestamp]
            Sorted list of holding dates
        """
        # Get range of portfolio dates
        min_date = portfolios['holding_start'].min()
        max_date = portfolios['holding_end'].max()
        
        # Get available return dates within this range
        stock_dates = pd.to_datetime(stock_data['date'])
        valid_dates = stock_dates[(stock_dates >= min_date) & (stock_dates <= max_date)]
        
        return sorted(valid_dates.unique())
    
    def _get_active_portfolios(self, 
                              portfolios: pd.DataFrame,
                              date: pd.Timestamp) -> pd.DataFrame:
        """
        Get portfolios active on a specific date.
        
        Parameters:
        -----------
        portfolios : pd.DataFrame
            All portfolio compositions
        date : pd.Timestamp
            Target date
            
        Returns:
        --------
        pd.DataFrame
            Active portfolios
        """
        return portfolios[
            (portfolios['holding_start'] <= date) &
            (portfolios['holding_end'] >= date)
        ].copy()
    
    def _aggregate_overlapping_portfolios(self, 
                                        active_portfolios: pd.DataFrame,
                                        date: pd.Timestamp,
                                        k: int,
                                        stock_id: str) -> pd.DataFrame:
        """
        Aggregate weights for overlapping portfolios on a specific date.
        
        Parameters:
        -----------
        active_portfolios : pd.DataFrame
            Portfolios active on the date
        date : pd.Timestamp
            Current date
        k : int
            Holding period (number of overlapping portfolios)
        stock_id : str
            Stock identifier column
            
        Returns:
        --------
        pd.DataFrame
            Aggregated portfolio holdings
        """
        # Each overlapping portfolio gets equal weight (1/K)
        portfolio_weight = 1.0 / k
        
        # Calculate adjusted weights
        active_portfolios['portfolio_weight'] = portfolio_weight
        active_portfolios['adjusted_weight'] = active_portfolios['weight'] * portfolio_weight
        
        # Aggregate by stock and portfolio number
        aggregated = active_portfolios.groupby([stock_id, 'portfolio_num']).agg({
            'adjusted_weight': 'sum',
            'momentum_score': 'mean',
            'formation_period': 'first',
            'holding_period': 'first',
            'portfolio_weight': 'sum'  # Total weight of this portfolio type
        }).reset_index()
        
        aggregated['date'] = date
        
        return aggregated
    
    def calculate_portfolio_turnover(self, 
                                   current_holdings: pd.DataFrame,
                                   new_holdings: pd.DataFrame,
                                   stock_id: str) -> float:
        """
        Calculate portfolio turnover between two periods.
        
        Turnover = 0.5 * sum(|w_new - w_old|) for all stocks
        
        Parameters:
        -----------
        current_holdings : pd.DataFrame
            Current portfolio holdings
        new_holdings : pd.DataFrame
            New portfolio holdings
        stock_id : str
            Stock identifier column
            
        Returns:
        --------
        float
            Portfolio turnover (0 to 1)
        """
        if current_holdings.empty or new_holdings.empty:
            return 1.0  # Complete turnover if one period is empty
        
        # Merge current and new weights
        current_weights = current_holdings[[stock_id, 'adjusted_weight']].copy()
        current_weights.columns = [stock_id, 'weight_old']
        
        new_weights = new_holdings[[stock_id, 'adjusted_weight']].copy()
        new_weights.columns = [stock_id, 'weight_new']
        
        # Full outer join to capture all stocks
        merged = pd.merge(current_weights, new_weights, on=stock_id, how='outer')
        merged = merged.fillna(0)  # Missing weights are zero
        
        # Calculate turnover
        weight_changes = abs(merged['weight_new'] - merged['weight_old'])
        turnover = 0.5 * weight_changes.sum()
        
        return turnover
    
    def calculate_strategy_turnover(self, 
                                  overlapping_holdings: pd.DataFrame,
                                  portfolio_num: int,
                                  stock_id: str) -> pd.DataFrame:
        """
        Calculate turnover statistics for a momentum strategy.
        
        Parameters:
        -----------
        overlapping_holdings : pd.DataFrame
            Overlapping portfolio holdings over time
        portfolio_num : int
            Portfolio number to analyze
        stock_id : str
            Stock identifier column
            
        Returns:
        --------
        pd.DataFrame
            Turnover statistics by date
        """
        portfolio_data = overlapping_holdings[
            overlapping_holdings['portfolio_num'] == portfolio_num
        ].sort_values('date')
        
        if len(portfolio_data) < 2:
            return pd.DataFrame()
        
        turnover_data = []
        dates = sorted(portfolio_data['date'].unique())
        
        for i in range(1, len(dates)):
            current_date = dates[i]
            previous_date = dates[i-1]
            
            current_holdings = portfolio_data[portfolio_data['date'] == current_date]
            previous_holdings = portfolio_data[portfolio_data['date'] == previous_date]
            
            turnover = self.calculate_portfolio_turnover(
                previous_holdings, current_holdings, stock_id
            )
            
            turnover_data.append({
                'date': current_date,
                'portfolio_num': portfolio_num,
                'turnover': turnover,
                'num_stocks_current': len(current_holdings),
                'num_stocks_previous': len(previous_holdings)
            })
        
        return pd.DataFrame(turnover_data)
    
    def rebalance_to_target_weights(self, 
                                  current_weights: pd.DataFrame,
                                  target_weights: pd.DataFrame,
                                  stock_id: str,
                                  transaction_cost_bps: float = config.TRANSACTION_COST_BPS) -> Dict:
        """
        Rebalance portfolio from current to target weights.
        
        Parameters:
        -----------
        current_weights : pd.DataFrame
            Current portfolio weights
        target_weights : pd.DataFrame
            Target portfolio weights
        stock_id : str
            Stock identifier column
        transaction_cost_bps : float
            Transaction costs in basis points
            
        Returns:
        --------
        Dict
            Rebalancing information including costs and trades
        """
        # Merge current and target weights
        current = current_weights[[stock_id, 'weight']].copy()
        current.columns = [stock_id, 'weight_current']
        
        target = target_weights[[stock_id, 'weight']].copy()
        target.columns = [stock_id, 'weight_target']
        
        rebalancing = pd.merge(current, target, on=stock_id, how='outer')
        rebalancing = rebalancing.fillna(0)
        
        # Calculate trades
        rebalancing['trade_size'] = rebalancing['weight_target'] - rebalancing['weight_current']
        rebalancing['abs_trade_size'] = abs(rebalancing['trade_size'])
        
        # Calculate transaction costs
        total_trade_value = rebalancing['abs_trade_size'].sum()
        transaction_costs = total_trade_value * (transaction_cost_bps / 10000)
        
        # Summary statistics
        result = {
            'total_trade_value': total_trade_value,
            'transaction_costs': transaction_costs,
            'num_trades': (rebalancing['abs_trade_size'] > 0.0001).sum(),
            'largest_trade': rebalancing['abs_trade_size'].max(),
            'turnover': total_trade_value / 2,  # Standard turnover definition
            'trades_detail': rebalancing
        }
        
        return result
    
    def simulate_portfolio_evolution(self, 
                                   initial_holdings: pd.DataFrame,
                                   stock_returns: pd.DataFrame,
                                   holding_dates: List[pd.Timestamp],
                                   stock_id: str) -> pd.DataFrame:
        """
        Simulate how portfolio weights evolve due to differential stock returns.
        
        Parameters:
        -----------
        initial_holdings : pd.DataFrame
            Initial portfolio holdings
        stock_returns : pd.DataFrame
            Stock return data
        holding_dates : List[pd.Timestamp]
            Dates to simulate
        stock_id : str
            Stock identifier column
            
        Returns:
        --------
        pd.DataFrame
            Portfolio evolution over time
        """
        evolution_data = []
        current_weights = initial_holdings.copy()
        
        for i, date in enumerate(holding_dates):
            if i == 0:
                # Initial period
                evolution_data.append({
                    'date': date,
                    'holdings': current_weights.copy()
                })
                continue
            
            # Get returns for this period
            period_returns = stock_returns[stock_returns['date'] == date]
            
            if period_returns.empty:
                continue
            
            # Update weights based on returns
            updated_weights = self._update_weights_with_returns(
                current_weights, period_returns, stock_id
            )
            
            evolution_data.append({
                'date': date,
                'holdings': updated_weights.copy()
            })
            
            current_weights = updated_weights
        
        return evolution_data
    
    def _update_weights_with_returns(self, 
                                   current_weights: pd.DataFrame,
                                   returns: pd.DataFrame,
                                   stock_id: str) -> pd.DataFrame:
        """
        Update portfolio weights based on stock returns.
        
        Parameters:
        -----------
        current_weights : pd.DataFrame
            Current portfolio weights
        returns : pd.DataFrame
            Stock returns for the period
        stock_id : str
            Stock identifier column
            
        Returns:
        --------
        pd.DataFrame
            Updated portfolio weights
        """
        # Merge weights with returns
        merged = pd.merge(
            current_weights[[stock_id, 'adjusted_weight']], 
            returns[[stock_id, 'ret']], 
            on=stock_id, 
            how='left'
        )
        
        # Fill missing returns with zero
        merged['ret'] = merged['ret'].fillna(0)
        
        # Calculate new values after returns
        merged['new_value'] = merged['adjusted_weight'] * (1 + merged['ret'])
        
        # Renormalize to sum to 1
        total_value = merged['new_value'].sum()
        
        if total_value > 0:
            merged['new_weight'] = merged['new_value'] / total_value
        else:
            merged['new_weight'] = 0
        
        # Update the weights
        updated_weights = current_weights.copy()
        updated_weights = pd.merge(
            updated_weights.drop('adjusted_weight', axis=1),
            merged[[stock_id, 'new_weight']],
            on=stock_id,
            how='left'
        )
        updated_weights.columns = [col if col != 'new_weight' else 'adjusted_weight' 
                                 for col in updated_weights.columns]
        
        return updated_weights
    
    def get_rebalancing_summary(self, 
                               overlapping_holdings: pd.DataFrame,
                               stock_id: str) -> Dict:
        """
        Get comprehensive rebalancing summary statistics.
        
        Parameters:
        -----------
        overlapping_holdings : pd.DataFrame
            Overlapping portfolio holdings
        stock_id : str
            Stock identifier column
            
        Returns:
        --------
        Dict
            Summary statistics
        """
        summary = {}
        
        # Calculate turnover for each portfolio
        for portfolio_num in overlapping_holdings['portfolio_num'].unique():
            turnover_data = self.calculate_strategy_turnover(
                overlapping_holdings, portfolio_num, stock_id
            )
            
            if not turnover_data.empty:
                summary[f'portfolio_{portfolio_num}'] = {
                    'mean_turnover': turnover_data['turnover'].mean(),
                    'median_turnover': turnover_data['turnover'].median(),
                    'std_turnover': turnover_data['turnover'].std(),
                    'min_turnover': turnover_data['turnover'].min(),
                    'max_turnover': turnover_data['turnover'].max(),
                    'mean_num_stocks': turnover_data['num_stocks_current'].mean()
                }
        
        # Overall statistics
        all_turnover = []
        for portfolio_num in overlapping_holdings['portfolio_num'].unique():
            turnover_data = self.calculate_strategy_turnover(
                overlapping_holdings, portfolio_num, stock_id
            )
            if not turnover_data.empty:
                all_turnover.extend(turnover_data['turnover'].tolist())
        
        if all_turnover:
            summary['overall'] = {
                'mean_turnover': np.mean(all_turnover),
                'median_turnover': np.median(all_turnover),
                'std_turnover': np.std(all_turnover),
                'observations': len(all_turnover)
            }
        
        return summary
    
    def _get_stock_id_column(self, data: pd.DataFrame) -> str:
        """Get the stock identifier column name."""
        if 'permno' in data.columns:
            return 'permno'
        elif 'ticker' in data.columns:
            return 'ticker'
        else:
            raise ValueError("No stock identifier column found (permno or ticker)")
    
    def save_rebalancing_data(self, 
                            data: pd.DataFrame,
                            filename: str = "rebalancing_data.csv") -> None:
        """Save rebalancing data to CSV."""
        data.to_csv(filename, index=False)
        logger.info(f"Saved rebalancing data to {filename}")

def main():
    """Example usage of PortfolioRebalancer."""
    from data_loader import DataLoader
    from data_cleaner import DataCleaner
    from momentum_calculator import MomentumCalculator
    from portfolio_builder import PortfolioBuilder
    
    # Load and prepare data
    loader = DataLoader()
    raw_data = loader.load_stock_data()
    loader.close_connection()
    
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_stock_data(raw_data)
    panel_data = cleaner.prepare_monthly_panel(cleaned_data)
    
    # Build portfolios
    momentum_calc = MomentumCalculator()
    momentum_scores = momentum_calc.calculate_momentum_scores(panel_data)
    
    portfolio_builder = PortfolioBuilder()
    portfolios = portfolio_builder.build_portfolios(momentum_scores, panel_data)
    
    # Create overlapping strategy
    rebalancer = PortfolioRebalancer()
    
    for (j, k), portfolio_data in portfolios.items():
        if not portfolio_data.empty:
            overlapping = rebalancer.create_overlapping_strategy(
                portfolio_data, panel_data, j, k
            )
            
            if not overlapping.empty:
                stock_id = rebalancer._get_stock_id_column(overlapping)
                summary = rebalancer.get_rebalancing_summary(overlapping, stock_id)
                
                print(f"\n({j},{k}) Rebalancing Summary:")
                for key, value in summary.get('overall', {}).items():
                    print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()