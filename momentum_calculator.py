"""
Momentum calculator module for Jegadeesh & Titman (1993) momentum strategy replication.
Calculates past returns for ranking periods and handles momentum score computation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings

import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOGGING_LEVEL))
logger = logging.getLogger(__name__)

class MomentumCalculator:
    """
    Class for calculating momentum scores and past returns for portfolio formation.
    Implements the Jegadeesh & Titman methodology for momentum calculation.
    """
    
    def __init__(self, skip_period: int = config.SKIP_PERIOD):
        """
        Initialize MomentumCalculator.
        
        Parameters:
        -----------
        skip_period : int
            Number of months to skip between formation and holding periods
            (to avoid microstructure effects like bid-ask bounce)
        """
        self.skip_period = skip_period
        self.momentum_data = {}
    
    def calculate_momentum_scores(self, 
                                data: pd.DataFrame,
                                formation_periods: List[int] = config.FORMATION_PERIODS) -> Dict[int, pd.DataFrame]:
        """
        Calculate momentum scores for all formation periods.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Monthly stock return data with columns: ['date', 'permno'/'ticker', 'ret', ...]
        formation_periods : List[int]
            List of formation periods (J months) to calculate momentum for
            
        Returns:
        --------
        Dict[int, pd.DataFrame]
            Dictionary mapping formation period to momentum scores DataFrame
        """
        logger.info(f"Calculating momentum scores for formation periods: {formation_periods}")
        
        # Identify stock identifier column
        stock_id = self._get_stock_id_column(data)
        
        # Ensure data is sorted
        data = data.sort_values([stock_id, 'date']).reset_index(drop=True)
        
        momentum_results = {}
        
        for j in formation_periods:
            logger.info(f"Calculating {j}-month momentum scores...")
            
            # Calculate cumulative returns over J months
            momentum_scores = self._calculate_cumulative_returns(data, j, stock_id)
            
            # Apply skip period
            if self.skip_period > 0:
                momentum_scores = self._apply_skip_period(momentum_scores, stock_id)
            
            # Store results
            momentum_results[j] = momentum_scores
            
            logger.info(f"Calculated momentum for {momentum_scores['date'].nunique()} months, "
                       f"{momentum_scores[stock_id].nunique()} unique stocks")
        
        self.momentum_data = momentum_results
        return momentum_results
    
    def _calculate_cumulative_returns(self, 
                                    data: pd.DataFrame, 
                                    formation_period: int,
                                    stock_id: str) -> pd.DataFrame:
        """
        Calculate cumulative returns over the formation period.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Stock return data
        formation_period : int
            Number of months for formation period (J)
        stock_id : str
            Stock identifier column name
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with momentum scores
        """
        momentum_scores = []
        
        # Get unique dates and stocks
        dates = sorted(data['date'].unique())
        
        for date in dates:
            # Find the start date for momentum calculation (J months before)
            start_date_idx = dates.index(date) - formation_period + 1
            
            if start_date_idx < 0:
                continue  # Not enough history
            
            start_date = dates[start_date_idx]
            
            # Get data for the formation period
            period_data = data[
                (data['date'] >= start_date) & 
                (data['date'] <= date)
            ].copy()
            
            # Calculate cumulative returns for each stock
            stock_momentum = self._calculate_stock_momentum(period_data, stock_id, formation_period)
            
            if not stock_momentum.empty:
                stock_momentum['formation_end_date'] = date
                stock_momentum['formation_period'] = formation_period
                momentum_scores.append(stock_momentum)
        
        if momentum_scores:
            return pd.concat(momentum_scores, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _calculate_stock_momentum(self, 
                                period_data: pd.DataFrame,
                                stock_id: str,
                                formation_period: int) -> pd.DataFrame:
        """
        Calculate momentum scores for individual stocks in a given period.
        
        Parameters:
        -----------
        period_data : pd.DataFrame
            Data for the formation period
        stock_id : str
            Stock identifier column
        formation_period : int
            Formation period length
            
        Returns:
        --------
        pd.DataFrame
            Stock momentum scores
        """
        stock_momentum = []
        
        for stock in period_data[stock_id].unique():
            stock_data = period_data[period_data[stock_id] == stock].sort_values('date')
            
            # Need at least minimum number of observations
            min_observations = max(formation_period // 2, 1)  # At least half the period
            
            if len(stock_data) < min_observations:
                continue
            
            # Calculate cumulative return (buy and hold)
            returns = stock_data['ret'].values
            
            # Handle missing values
            valid_returns = returns[~np.isnan(returns)]
            
            if len(valid_returns) < min_observations:
                continue
            
            # Calculate cumulative return: (1+r1)*(1+r2)*...*(1+rn) - 1
            cumulative_return = np.prod(1 + valid_returns) - 1
            
            # Alternative: compound monthly returns
            # cumulative_return = (1 + pd.Series(valid_returns)).prod() - 1
            
            stock_momentum.append({
                stock_id: stock,
                'momentum_score': cumulative_return,
                'num_observations': len(valid_returns),
                'formation_start_date': stock_data['date'].min(),
                'formation_end_date': stock_data['date'].max()
            })
        
        return pd.DataFrame(stock_momentum)
    
    def _apply_skip_period(self, momentum_data: pd.DataFrame, stock_id: str) -> pd.DataFrame:
        """
        Apply skip period between formation and holding periods.
        
        Parameters:
        -----------
        momentum_data : pd.DataFrame
            Momentum scores
        stock_id : str
            Stock identifier column
            
        Returns:
        --------
        pd.DataFrame
            Momentum data with adjusted dates for skip period
        """
        if self.skip_period == 0:
            return momentum_data
        
        logger.info(f"Applying {self.skip_period}-month skip period")
        
        # Shift formation end date forward by skip period
        momentum_data = momentum_data.copy()
        
        # Convert to datetime if not already
        momentum_data['formation_end_date'] = pd.to_datetime(momentum_data['formation_end_date'])
        
        # Add skip period (approximate months by adding days)
        momentum_data['portfolio_start_date'] = momentum_data['formation_end_date'] + pd.DateOffset(months=self.skip_period)
        
        return momentum_data
    
    def get_momentum_rankings(self, 
                            momentum_data: pd.DataFrame,
                            date: datetime,
                            stock_id: str,
                            num_portfolios: int = config.NUM_PORTFOLIOS) -> pd.DataFrame:
        """
        Get momentum rankings for portfolio formation at a specific date.
        
        Parameters:
        -----------
        momentum_data : pd.DataFrame
            Momentum scores data
        date : datetime
            Portfolio formation date
        stock_id : str
            Stock identifier column
        num_portfolios : int
            Number of momentum portfolios to form
            
        Returns:
        --------
        pd.DataFrame
            Rankings with portfolio assignments
        """
        # Get momentum scores for the specific date
        date_data = momentum_data[
            momentum_data['formation_end_date'] == date
        ].copy()
        
        if date_data.empty:
            return pd.DataFrame()
        
        # Remove stocks with missing momentum scores
        date_data = date_data.dropna(subset=['momentum_score'])
        
        if len(date_data) < num_portfolios:
            logger.warning(f"Too few stocks ({len(date_data)}) for {num_portfolios} portfolios on {date}")
            return pd.DataFrame()
        
        # Rank stocks by momentum score (ascending: worst to best)
        date_data = date_data.sort_values('momentum_score').reset_index(drop=True)
        
        # Assign portfolio numbers (1 = lowest momentum, 10 = highest momentum)
        date_data['momentum_rank'] = range(1, len(date_data) + 1)
        
        # Create portfolio assignments
        stocks_per_portfolio = len(date_data) // num_portfolios
        
        portfolio_assignments = []
        
        for portfolio in range(1, num_portfolios + 1):
            start_idx = (portfolio - 1) * stocks_per_portfolio
            
            if portfolio == num_portfolios:
                # Last portfolio gets remaining stocks
                end_idx = len(date_data)
            else:
                end_idx = start_idx + stocks_per_portfolio
            
            portfolio_stocks = date_data.iloc[start_idx:end_idx].copy()
            portfolio_stocks['portfolio'] = portfolio
            portfolio_assignments.append(portfolio_stocks)
        
        if portfolio_assignments:
            result = pd.concat(portfolio_assignments, ignore_index=True)
            
            # Add portfolio labels
            result['portfolio_label'] = result['portfolio'].apply(
                lambda x: 'Loser' if x == 1 else ('Winner' if x == num_portfolios else f'P{x}')
            )
            
            return result
        else:
            return pd.DataFrame()
    
    def calculate_momentum_percentiles(self, 
                                     momentum_data: pd.DataFrame,
                                     percentiles: List[float] = [10, 25, 50, 75, 90]) -> pd.DataFrame:
        """
        Calculate momentum score percentiles by date.
        
        Parameters:
        -----------
        momentum_data : pd.DataFrame
            Momentum scores
        percentiles : List[float]
            Percentiles to calculate
            
        Returns:
        --------
        pd.DataFrame
            Momentum percentiles by date
        """
        percentile_data = []
        
        for date in momentum_data['formation_end_date'].unique():
            date_data = momentum_data[momentum_data['formation_end_date'] == date]
            
            if date_data.empty:
                continue
            
            momentum_scores = date_data['momentum_score'].dropna()
            
            if len(momentum_scores) == 0:
                continue
            
            percentile_values = np.percentile(momentum_scores, percentiles)
            
            row = {'date': date, 'num_stocks': len(momentum_scores)}
            for pct, val in zip(percentiles, percentile_values):
                row[f'p{pct}'] = val
            
            percentile_data.append(row)
        
        return pd.DataFrame(percentile_data)
    
    def get_momentum_summary_statistics(self, momentum_data: pd.DataFrame) -> Dict:
        """
        Get summary statistics for momentum scores.
        
        Parameters:
        -----------
        momentum_data : pd.DataFrame
            Momentum scores
            
        Returns:
        --------
        Dict
            Summary statistics
        """
        momentum_scores = momentum_data['momentum_score'].dropna()
        
        if len(momentum_scores) == 0:
            return {}
        
        stats = {
            'count': len(momentum_scores),
            'mean': momentum_scores.mean(),
            'median': momentum_scores.median(),
            'std': momentum_scores.std(),
            'min': momentum_scores.min(),
            'max': momentum_scores.max(),
            'skewness': momentum_scores.skew(),
            'kurtosis': momentum_scores.kurtosis(),
            'positive_momentum_pct': (momentum_scores > 0).mean() * 100,
            'extreme_winners_pct': (momentum_scores > momentum_scores.quantile(0.9)).mean() * 100,
            'extreme_losers_pct': (momentum_scores < momentum_scores.quantile(0.1)).mean() * 100
        }
        
        return stats
    
    def _get_stock_id_column(self, data: pd.DataFrame) -> str:
        """Get the stock identifier column name."""
        if 'permno' in data.columns:
            return 'permno'
        elif 'ticker' in data.columns:
            return 'ticker'
        else:
            raise ValueError("No stock identifier column found (permno or ticker)")
    
    def save_momentum_data(self, filename: str = "momentum_scores.csv") -> None:
        """Save momentum data to CSV."""
        if not self.momentum_data:
            logger.warning("No momentum data to save")
            return
        
        for formation_period, data in self.momentum_data.items():
            filepath = f"momentum_{formation_period}m_{filename}"
            data.to_csv(filepath, index=False)
            logger.info(f"Saved {formation_period}-month momentum data to {filepath}")

def main():
    """Example usage of MomentumCalculator."""
    from data_loader import DataLoader
    from data_cleaner import DataCleaner
    
    # Load and clean data
    loader = DataLoader()
    raw_data = loader.load_stock_data()
    loader.close_connection()
    
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_stock_data(raw_data)
    panel_data = cleaner.prepare_monthly_panel(cleaned_data)
    
    # Calculate momentum scores
    momentum_calc = MomentumCalculator()
    momentum_scores = momentum_calc.calculate_momentum_scores(panel_data)
    
    # Display summary for each formation period
    for formation_period, data in momentum_scores.items():
        print(f"\n{formation_period}-Month Momentum Summary:")
        stats = momentum_calc.get_momentum_summary_statistics(data)
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")
    
    # Save results
    momentum_calc.save_momentum_data()

if __name__ == "__main__":
    main()