"""
Portfolio builder module for Jegadeesh & Titman (1993) momentum strategy replication.
Forms winner and loser portfolios based on momentum rankings.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings

import config
from momentum_calculator import MomentumCalculator

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOGGING_LEVEL))
logger = logging.getLogger(__name__)

class PortfolioBuilder:
    """
    Class for building momentum portfolios based on past return rankings.
    Implements the Jegadeesh & Titman portfolio formation methodology.
    """
    
    def __init__(self, 
                 num_portfolios: int = config.NUM_PORTFOLIOS,
                 weighting_scheme: str = config.PORTFOLIO_WEIGHTING):
        """
        Initialize PortfolioBuilder.
        
        Parameters:
        -----------
        num_portfolios : int
            Number of momentum portfolios (typically 10 for deciles)
        weighting_scheme : str
            Portfolio weighting scheme ('equal' or 'value')
        """
        self.num_portfolios = num_portfolios
        self.weighting_scheme = weighting_scheme
        self.portfolios = {}
    
    def build_portfolios(self, 
                        momentum_data: Dict[int, pd.DataFrame],
                        stock_data: pd.DataFrame,
                        formation_periods: List[int] = config.FORMATION_PERIODS,
                        holding_periods: List[int] = config.HOLDING_PERIODS) -> Dict[Tuple[int, int], pd.DataFrame]:
        """
        Build momentum portfolios for all (J,K) combinations.
        
        Parameters:
        -----------
        momentum_data : Dict[int, pd.DataFrame]
            Momentum scores by formation period
        stock_data : pd.DataFrame
            Stock return and price data
        formation_periods : List[int]
            Formation periods (J months)
        holding_periods : List[int]
            Holding periods (K months)
            
        Returns:
        --------
        Dict[Tuple[int, int], pd.DataFrame]
            Portfolio compositions for each (J,K) strategy
        """
        logger.info(f"Building portfolios for {len(formation_periods)} formation × {len(holding_periods)} holding period combinations")
        
        stock_id = self._get_stock_id_column(stock_data)
        
        all_portfolios = {}
        
        for j in formation_periods:
            if j not in momentum_data:
                logger.warning(f"No momentum data available for {j}-month formation period")
                continue
            
            for k in holding_periods:
                logger.info(f"Building portfolios for ({j},{k}) strategy...")
                
                # Build portfolios for this (J,K) combination
                jk_portfolios = self._build_jk_portfolios(
                    momentum_data[j], stock_data, j, k, stock_id
                )
                
                all_portfolios[(j, k)] = jk_portfolios
                
                if not jk_portfolios.empty:
                    logger.info(f"Built ({j},{k}) portfolios: {jk_portfolios['date'].nunique()} months, "
                              f"{jk_portfolios[stock_id].nunique()} unique stocks")
        
        self.portfolios = all_portfolios
        return all_portfolios
    
    def _build_jk_portfolios(self, 
                           momentum_data: pd.DataFrame,
                           stock_data: pd.DataFrame,
                           j: int, k: int,
                           stock_id: str) -> pd.DataFrame:
        """
        Build portfolios for a specific (J,K) combination.
        
        Parameters:
        -----------
        momentum_data : pd.DataFrame
            Momentum scores for formation period J
        stock_data : pd.DataFrame
            Stock return and price data
        j : int
            Formation period (months)
        k : int
            Holding period (months)
        stock_id : str
            Stock identifier column
            
        Returns:
        --------
        pd.DataFrame
            Portfolio compositions with holdings and weights
        """
        portfolio_compositions = []
        
        # Get unique formation dates
        formation_dates = sorted(momentum_data['formation_end_date'].unique())
        
        for formation_date in formation_dates:
            # Get momentum rankings for this date
            momentum_calc = MomentumCalculator()
            rankings = momentum_calc.get_momentum_rankings(
                momentum_data, formation_date, stock_id, self.num_portfolios
            )
            
            if rankings.empty:
                continue
            
            # Calculate holding period end date
            holding_start = formation_date + pd.DateOffset(months=config.SKIP_PERIOD)
            holding_end = holding_start + pd.DateOffset(months=k)
            
            # Create portfolio compositions
            for portfolio_num in range(1, self.num_portfolios + 1):
                portfolio_stocks = rankings[rankings['portfolio'] == portfolio_num].copy()
                
                if portfolio_stocks.empty:
                    continue
                
                # Calculate weights for this portfolio
                weights = self._calculate_portfolio_weights(
                    portfolio_stocks, stock_data, formation_date, stock_id
                )
                
                portfolio_stocks = portfolio_stocks.merge(weights, on=stock_id, how='left')
                
                # Add portfolio metadata
                portfolio_stocks['formation_period'] = j
                portfolio_stocks['holding_period'] = k
                portfolio_stocks['formation_date'] = formation_date
                portfolio_stocks['holding_start'] = holding_start
                portfolio_stocks['holding_end'] = holding_end
                portfolio_stocks['portfolio_num'] = portfolio_num
                
                portfolio_compositions.append(portfolio_stocks)
        
        if portfolio_compositions:
            return pd.concat(portfolio_compositions, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _calculate_portfolio_weights(self, 
                                   portfolio_stocks: pd.DataFrame,
                                   stock_data: pd.DataFrame,
                                   formation_date: datetime,
                                   stock_id: str) -> pd.DataFrame:
        """
        Calculate portfolio weights based on the weighting scheme.
        
        Parameters:
        -----------
        portfolio_stocks : pd.DataFrame
            Stocks in the portfolio
        stock_data : pd.DataFrame
            Full stock dataset
        formation_date : datetime
            Portfolio formation date
        stock_id : str
            Stock identifier column
            
        Returns:
        --------
        pd.DataFrame
            Weights for portfolio stocks
        """
        if self.weighting_scheme == 'equal':
            # Equal weighting
            num_stocks = len(portfolio_stocks)
            weights = pd.DataFrame({
                stock_id: portfolio_stocks[stock_id],
                'weight': 1.0 / num_stocks
            })
            
        elif self.weighting_scheme == 'value':
            # Value weighting based on market capitalization
            weights = self._calculate_value_weights(
                portfolio_stocks, stock_data, formation_date, stock_id
            )
            
        else:
            raise ValueError(f"Unknown weighting scheme: {self.weighting_scheme}")
        
        return weights
    
    def _calculate_value_weights(self, 
                               portfolio_stocks: pd.DataFrame,
                               stock_data: pd.DataFrame,
                               formation_date: datetime,
                               stock_id: str) -> pd.DataFrame:
        """
        Calculate value weights based on market capitalization.
        
        Parameters:
        -----------
        portfolio_stocks : pd.DataFrame
            Stocks in the portfolio
        stock_data : pd.DataFrame
            Full stock dataset
        formation_date : datetime
            Portfolio formation date
        stock_id : str
            Stock identifier column
            
        Returns:
        --------
        pd.DataFrame
            Value weights for portfolio stocks
        """
        # Get market cap data for formation date
        formation_data = stock_data[
            (stock_data['date'] == formation_date) & 
            (stock_data[stock_id].isin(portfolio_stocks[stock_id]))
        ].copy()
        
        if 'market_cap' not in formation_data.columns:
            logger.warning("Market cap not available, using equal weights")
            num_stocks = len(portfolio_stocks)
            return pd.DataFrame({
                stock_id: portfolio_stocks[stock_id],
                'weight': 1.0 / num_stocks
            })
        
        # Calculate value weights
        formation_data = formation_data.dropna(subset=['market_cap'])
        
        if formation_data.empty:
            # Fall back to equal weights
            num_stocks = len(portfolio_stocks)
            return pd.DataFrame({
                stock_id: portfolio_stocks[stock_id],
                'weight': 1.0 / num_stocks
            })
        
        total_market_cap = formation_data['market_cap'].sum()
        formation_data['weight'] = formation_data['market_cap'] / total_market_cap
        
        # Ensure all portfolio stocks have weights (missing ones get zero weight)
        weights = pd.DataFrame({stock_id: portfolio_stocks[stock_id]})
        weights = weights.merge(
            formation_data[[stock_id, 'weight']], 
            on=stock_id, 
            how='left'
        )
        weights['weight'] = weights['weight'].fillna(0)
        
        # Renormalize to sum to 1
        total_weight = weights['weight'].sum()
        if total_weight > 0:
            weights['weight'] = weights['weight'] / total_weight
        else:
            # If all weights are zero, use equal weights
            weights['weight'] = 1.0 / len(weights)
        
        return weights
    
    def get_portfolio_composition(self, 
                                formation_date: datetime,
                                j: int, k: int,
                                portfolio_num: int) -> pd.DataFrame:
        """
        Get portfolio composition for a specific date and strategy.
        
        Parameters:
        -----------
        formation_date : datetime
            Portfolio formation date
        j : int
            Formation period
        k : int
            Holding period
        portfolio_num : int
            Portfolio number (1 to num_portfolios)
            
        Returns:
        --------
        pd.DataFrame
            Portfolio composition
        """
        if (j, k) not in self.portfolios:
            logger.warning(f"No portfolios available for ({j},{k}) strategy")
            return pd.DataFrame()
        
        portfolio_data = self.portfolios[(j, k)]
        
        composition = portfolio_data[
            (portfolio_data['formation_date'] == formation_date) &
            (portfolio_data['portfolio_num'] == portfolio_num)
        ].copy()
        
        return composition
    
    def get_winner_loser_portfolios(self, 
                                  formation_date: datetime,
                                  j: int, k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get winner and loser portfolios for a specific strategy.
        
        Parameters:
        -----------
        formation_date : datetime
            Portfolio formation date
        j : int
            Formation period
        k : int
            Holding period
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Winner portfolio, Loser portfolio
        """
        winner_portfolio = self.get_portfolio_composition(formation_date, j, k, self.num_portfolios)
        loser_portfolio = self.get_portfolio_composition(formation_date, j, k, 1)
        
        return winner_portfolio, loser_portfolio
    
    def get_portfolio_statistics(self, j: int, k: int) -> Dict:
        """
        Get statistics for portfolios in a (J,K) strategy.
        
        Parameters:
        -----------
        j : int
            Formation period
        k : int
            Holding period
            
        Returns:
        --------
        Dict
            Portfolio statistics
        """
        if (j, k) not in self.portfolios:
            return {}
        
        portfolio_data = self.portfolios[(j, k)]
        stock_id = self._get_stock_id_column(portfolio_data)
        
        stats = {}
        
        # Overall statistics
        stats['total_observations'] = len(portfolio_data)
        stats['unique_stocks'] = portfolio_data[stock_id].nunique()
        stats['formation_dates'] = portfolio_data['formation_date'].nunique()
        
        # Portfolio-specific statistics
        for portfolio_num in range(1, self.num_portfolios + 1):
            portfolio_subset = portfolio_data[portfolio_data['portfolio_num'] == portfolio_num]
            
            if not portfolio_subset.empty:
                stats[f'portfolio_{portfolio_num}_stocks_per_period'] = (
                    portfolio_subset.groupby('formation_date').size().mean()
                )
                stats[f'portfolio_{portfolio_num}_avg_momentum'] = (
                    portfolio_subset['momentum_score'].mean()
                )
                stats[f'portfolio_{portfolio_num}_avg_weight'] = (
                    portfolio_subset['weight'].mean()
                )
        
        # Winner-Loser spread statistics
        winner_data = portfolio_data[portfolio_data['portfolio_num'] == self.num_portfolios]
        loser_data = portfolio_data[portfolio_data['portfolio_num'] == 1]
        
        if not winner_data.empty and not loser_data.empty:
            stats['winner_avg_momentum'] = winner_data['momentum_score'].mean()
            stats['loser_avg_momentum'] = loser_data['momentum_score'].mean()
            stats['momentum_spread'] = stats['winner_avg_momentum'] - stats['loser_avg_momentum']
        
        return stats
    
    def create_overlapping_portfolios(self, 
                                    j: int, k: int,
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None) -> Dict[int, pd.DataFrame]:
        """
        Create overlapping portfolios following JT methodology.
        
        In the JT approach, portfolios are formed every month but held for K months,
        creating overlapping holdings that need to be properly weighted.
        
        Parameters:
        -----------
        j : int
            Formation period
        k : int
            Holding period
        start_date : datetime, optional
            Start date for analysis
        end_date : datetime, optional
            End date for analysis
            
        Returns:
        --------
        Dict[int, pd.DataFrame]
            Overlapping portfolio holdings by month
        """
        if (j, k) not in self.portfolios:
            logger.warning(f"No portfolios available for ({j},{k}) strategy")
            return {}
        
        portfolio_data = self.portfolios[(j, k)]
        
        if start_date is None:
            start_date = portfolio_data['holding_start'].min()
        if end_date is None:
            end_date = portfolio_data['holding_end'].max()
        
        # Generate monthly dates
        monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        overlapping_portfolios = {}
        
        for date in monthly_dates:
            # Find all portfolios active on this date
            active_portfolios = portfolio_data[
                (portfolio_data['holding_start'] <= date) &
                (portfolio_data['holding_end'] >= date)
            ].copy()
            
            if active_portfolios.empty:
                continue
            
            # Group by portfolio number and aggregate weights
            monthly_holdings = self._aggregate_overlapping_weights(active_portfolios, date, k)
            
            overlapping_portfolios[date] = monthly_holdings
        
        return overlapping_portfolios
    
    def _aggregate_overlapping_weights(self, 
                                     active_portfolios: pd.DataFrame,
                                     date: datetime,
                                     k: int) -> pd.DataFrame:
        """
        Aggregate weights for overlapping portfolio holdings.
        
        Parameters:
        -----------
        active_portfolios : pd.DataFrame
            Active portfolio holdings
        date : datetime
            Current date
        k : int
            Holding period
            
        Returns:
        --------
        pd.DataFrame
            Aggregated portfolio holdings
        """
        stock_id = self._get_stock_id_column(active_portfolios)
        
        # Each active portfolio gets weight 1/K (since K portfolios overlap)
        portfolio_weight = 1.0 / k
        
        # Adjust individual stock weights
        active_portfolios['adjusted_weight'] = active_portfolios['weight'] * portfolio_weight
        
        # Aggregate by stock and portfolio number
        aggregated = active_portfolios.groupby([stock_id, 'portfolio_num']).agg({
            'adjusted_weight': 'sum',
            'momentum_score': 'mean',  # Average momentum score
            'formation_period': 'first',
            'holding_period': 'first'
        }).reset_index()
        
        aggregated['date'] = date
        
        return aggregated
    
    def _get_stock_id_column(self, data: pd.DataFrame) -> str:
        """Get the stock identifier column name."""
        if 'permno' in data.columns:
            return 'permno'
        elif 'ticker' in data.columns:
            return 'ticker'
        else:
            raise ValueError("No stock identifier column found (permno or ticker)")
    
    def save_portfolios(self, filename_prefix: str = "portfolios") -> None:
        """Save portfolio data to CSV files."""
        if not self.portfolios:
            logger.warning("No portfolio data to save")
            return
        
        for (j, k), portfolio_data in self.portfolios.items():
            filename = f"{filename_prefix}_{j}_{k}.csv"
            portfolio_data.to_csv(filename, index=False)
            logger.info(f"Saved ({j},{k}) portfolio data to {filename}")

def main():
    """Example usage of PortfolioBuilder."""
    from data_loader import DataLoader
    from data_cleaner import DataCleaner
    from momentum_calculator import MomentumCalculator
    
    # Load and prepare data
    loader = DataLoader()
    raw_data = loader.load_stock_data()
    loader.close_connection()
    
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_stock_data(raw_data)
    panel_data = cleaner.prepare_monthly_panel(cleaned_data)
    
    # Calculate momentum scores
    momentum_calc = MomentumCalculator()
    momentum_scores = momentum_calc.calculate_momentum_scores(panel_data)
    
    # Build portfolios
    portfolio_builder = PortfolioBuilder()
    portfolios = portfolio_builder.build_portfolios(momentum_scores, panel_data)
    
    # Display portfolio statistics
    for (j, k), portfolio_data in portfolios.items():
        print(f"\n({j},{k}) Portfolio Statistics:")
        stats = portfolio_builder.get_portfolio_statistics(j, k)
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Save portfolios
    portfolio_builder.save_portfolios()

if __name__ == "__main__":
    main()