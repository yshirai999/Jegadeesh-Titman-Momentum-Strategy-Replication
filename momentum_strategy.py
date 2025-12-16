"""
Momentum strategy module for Jegadeesh & Titman (1993) replication.
Main strategy implementation class that coordinates all components.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import warnings

import config
from data_loader import DataLoader
from data_cleaner import DataCleaner
from data_validator import DataValidator
from momentum_calculator import MomentumCalculator
from portfolio_builder import PortfolioBuilder
from rebalancer import PortfolioRebalancer
from returns_calculator import ReturnsCalculator
from utils import get_jk_combinations, ensure_directory_exists

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOGGING_LEVEL))
logger = logging.getLogger(__name__)

class MomentumStrategy:
    """
    Main momentum strategy class implementing Jegadeesh & Titman (1993) methodology.
    Coordinates data loading, portfolio construction, and return calculation.
    """
    
    def __init__(self, 
                 formation_periods: List[int] = config.FORMATION_PERIODS,
                 holding_periods: List[int] = config.HOLDING_PERIODS,
                 num_portfolios: int = config.NUM_PORTFOLIOS,
                 weighting_scheme: str = config.PORTFOLIO_WEIGHTING):
        """
        Initialize MomentumStrategy.
        
        Parameters:
        -----------
        formation_periods : List[int]
            Formation periods (J months)
        holding_periods : List[int]
            Holding periods (K months)
        num_portfolios : int
            Number of momentum portfolios
        weighting_scheme : str
            Portfolio weighting scheme ('equal' or 'value')
        """
        self.formation_periods = formation_periods
        self.holding_periods = holding_periods
        self.num_portfolios = num_portfolios
        self.weighting_scheme = weighting_scheme
        
        # Initialize components
        self.data_loader = None
        self.data_cleaner = None
        self.data_validator = None
        self.momentum_calculator = None
        self.portfolio_builder = None
        self.rebalancer = None
        self.returns_calculator = None
        
        # Data storage
        self.raw_data = None
        self.cleaned_data = None
        self.market_data = None
        self.momentum_scores = {}
        self.portfolios = {}
        self.overlapping_portfolios = {}
        self.strategy_returns = {}
        
        # Results
        self.results = {}
    
    def load_and_prepare_data(self, 
                            data_source: str = config.DATA_SOURCE,
                            validate_data: bool = True) -> None:
        """
        Load and prepare data for momentum strategy analysis.
        
        Parameters:
        -----------
        data_source : str
            Data source ('wrds', 'yahoo', 'local_csv')
        validate_data : bool
            Whether to run data validation
        """
        logger.info("Loading and preparing data...")
        
        # Initialize data loader
        self.data_loader = DataLoader(data_source)
        
        # Load raw data
        self.raw_data = self.data_loader.load_stock_data()
        self.market_data = self.data_loader.load_market_data()
        
        # Clean data
        self.data_cleaner = DataCleaner()
        self.cleaned_data = self.data_cleaner.clean_stock_data(self.raw_data)
        self.cleaned_data = self.data_cleaner.prepare_monthly_panel(self.cleaned_data)
        
        # Validate data
        if validate_data:
            self.data_validator = DataValidator()
            validation_results = self.data_validator.validate_data(
                self.cleaned_data, self.market_data
            )
            
            if not self.data_validator.is_data_valid():
                logger.warning("Data validation found errors. Proceeding with caution.")
        
        # Close data loader connection
        if self.data_loader:
            self.data_loader.close_connection()
        
        logger.info(f"Data preparation complete: {len(self.cleaned_data):,} observations, "
                   f"{self.cleaned_data['permno'].nunique() if 'permno' in self.cleaned_data.columns else self.cleaned_data['ticker'].nunique()} stocks")
    
    def calculate_momentum_scores(self) -> None:
        """
        Calculate momentum scores for all formation periods.
        """
        logger.info("Calculating momentum scores...")
        
        if self.cleaned_data is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        self.momentum_calculator = MomentumCalculator()
        self.momentum_scores = self.momentum_calculator.calculate_momentum_scores(
            self.cleaned_data, self.formation_periods
        )
        
        logger.info(f"Momentum scores calculated for {len(self.momentum_scores)} formation periods")
    
    def build_portfolios(self) -> None:
        """
        Build momentum portfolios for all (J,K) combinations.
        """
        logger.info("Building momentum portfolios...")
        
        if not self.momentum_scores:
            raise ValueError("Momentum scores not calculated. Call calculate_momentum_scores() first.")
        
        self.portfolio_builder = PortfolioBuilder(
            num_portfolios=self.num_portfolios,
            weighting_scheme=self.weighting_scheme
        )
        
        self.portfolios = self.portfolio_builder.build_portfolios(
            self.momentum_scores,
            self.cleaned_data,
            self.formation_periods,
            self.holding_periods
        )
        
        logger.info(f"Portfolios built for {len(self.portfolios)} (J,K) combinations")
    
    def create_overlapping_strategies(self) -> None:
        """
        Create overlapping portfolio strategies.
        """
        logger.info("Creating overlapping portfolio strategies...")
        
        if not self.portfolios:
            raise ValueError("Portfolios not built. Call build_portfolios() first.")
        
        self.rebalancer = PortfolioRebalancer()
        
        for (j, k), portfolio_data in self.portfolios.items():
            if not portfolio_data.empty:
                overlapping = self.rebalancer.create_overlapping_strategy(
                    portfolio_data, self.cleaned_data, j, k
                )
                self.overlapping_portfolios[(j, k)] = overlapping
        
        logger.info(f"Overlapping strategies created for {len(self.overlapping_portfolios)} (J,K) combinations")
    
    def calculate_strategy_returns(self) -> None:
        """
        Calculate returns for all momentum strategies.
        """
        logger.info("Calculating strategy returns...")
        
        if not self.overlapping_portfolios:
            raise ValueError("Overlapping portfolios not created. Call create_overlapping_strategies() first.")
        
        self.returns_calculator = ReturnsCalculator()
        
        for (j, k), overlapping_data in self.overlapping_portfolios.items():
            if not overlapping_data.empty:
                strategy_returns = self.returns_calculator.calculate_portfolio_returns(
                    overlapping_data, self.cleaned_data, j, k
                )
                self.strategy_returns[(j, k)] = strategy_returns
        
        logger.info(f"Strategy returns calculated for {len(self.strategy_returns)} (J,K) combinations")
    
    def run_full_analysis(self, 
                         data_source: str = config.DATA_SOURCE,
                         validate_data: bool = True) -> Dict:
        """
        Run complete momentum strategy analysis.
        
        Parameters:
        -----------
        data_source : str
            Data source to use
        validate_data : bool
            Whether to validate data
            
        Returns:
        --------
        Dict
            Complete analysis results
        """
        logger.info("Starting full momentum strategy analysis...")
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data(data_source, validate_data)
            
            # Step 2: Calculate momentum scores
            self.calculate_momentum_scores()
            
            # Step 3: Build portfolios
            self.build_portfolios()
            
            # Step 4: Create overlapping strategies
            self.create_overlapping_strategies()
            
            # Step 5: Calculate returns
            self.calculate_strategy_returns()
            
            # Step 6: Compile results
            self.results = self._compile_results()
            
            logger.info("Full momentum strategy analysis completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in momentum strategy analysis: {e}")
            raise
    
    def _compile_results(self) -> Dict:
        """
        Compile all analysis results into a structured format.
        
        Returns:
        --------
        Dict
            Compiled results
        """
        results = {
            'data_summary': self._get_data_summary(),
            'momentum_summary': self._get_momentum_summary(),
            'portfolio_summary': self._get_portfolio_summary(),
            'strategy_returns': self.strategy_returns,
            'performance_metrics': self._calculate_performance_metrics()
        }
        
        return results
    
    def _get_data_summary(self) -> Dict:
        """
        Get summary of data used in analysis.
        """
        if self.cleaned_data is None:
            return {}
        
        stock_id = 'permno' if 'permno' in self.cleaned_data.columns else 'ticker'
        
        return {
            'total_observations': len(self.cleaned_data),
            'unique_stocks': self.cleaned_data[stock_id].nunique(),
            'date_range': {
                'start': self.cleaned_data['date'].min(),
                'end': self.cleaned_data['date'].max()
            },
            'monthly_observations': self.cleaned_data['date'].nunique(),
            'avg_stocks_per_month': len(self.cleaned_data) / self.cleaned_data['date'].nunique()
        }
    
    def _get_momentum_summary(self) -> Dict:
        """
        Get summary of momentum calculations.
        """
        summary = {}
        
        for j, momentum_data in self.momentum_scores.items():
            if not momentum_data.empty:
                stats = self.momentum_calculator.get_momentum_summary_statistics(momentum_data)
                summary[f'{j}_month'] = stats
        
        return summary
    
    def _get_portfolio_summary(self) -> Dict:
        """
        Get summary of portfolio construction.
        """
        summary = {}
        
        for (j, k), portfolio_data in self.portfolios.items():
            if not portfolio_data.empty:
                stats = self.portfolio_builder.get_portfolio_statistics(j, k)
                summary[f'({j},{k})'] = stats
        
        return summary
    
    def _calculate_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics for all strategies.
        """
        if not self.returns_calculator:
            return {}
        
        performance = {}
        
        for (j, k), returns_data in self.strategy_returns.items():
            if not returns_data.empty:
                metrics = self.returns_calculator.calculate_performance_metrics(
                    returns_data, self.market_data
                )
                performance[f'({j},{k})'] = metrics
        
        return performance
    
    def get_winner_loser_spreads(self) -> pd.DataFrame:
        """
        Calculate winner-loser portfolio spreads for all strategies.
        
        Returns:
        --------
        pd.DataFrame
            Winner-loser spreads with statistical significance
        """
        if not self.strategy_returns:
            return pd.DataFrame()
        
        spreads = []
        
        for (j, k), returns_data in self.strategy_returns.items():
            if not returns_data.empty:
                winner_returns = returns_data[returns_data['portfolio_num'] == self.num_portfolios]
                loser_returns = returns_data[returns_data['portfolio_num'] == 1]
                
                if not winner_returns.empty and not loser_returns.empty:
                    # Merge on date
                    merged = pd.merge(
                        winner_returns[['date', 'portfolio_return']], 
                        loser_returns[['date', 'portfolio_return']], 
                        on='date', suffixes=('_winner', '_loser')
                    )
                    
                    if not merged.empty:
                        merged['spread'] = merged['portfolio_return_winner'] - merged['portfolio_return_loser']
                        
                        spread_stats = {
                            'formation_period': j,
                            'holding_period': k,
                            'mean_spread': merged['spread'].mean(),
                            'std_spread': merged['spread'].std(),
                            't_stat': merged['spread'].mean() / (merged['spread'].std() / np.sqrt(len(merged))),
                            'observations': len(merged)
                        }
                        
                        spreads.append(spread_stats)
        
        return pd.DataFrame(spreads)
    
    def save_results(self, output_dir: str = config.RESULTS_DIR) -> None:
        """
        Save all results to files.
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        ensure_directory_exists(output_dir)
        
        # Save strategy returns
        for (j, k), returns_data in self.strategy_returns.items():
            if not returns_data.empty:
                filename = f"strategy_returns_{j}_{k}.csv"
                filepath = f"{output_dir}/{filename}"
                returns_data.to_csv(filepath, index=False)
        
        # Save winner-loser spreads
        spreads = self.get_winner_loser_spreads()
        if not spreads.empty:
            spreads.to_csv(f"{output_dir}/winner_loser_spreads.csv", index=False)
        
        # Save compiled results
        import json
        with open(f"{output_dir}/analysis_results.json", 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            results_json = self._convert_numpy_types(self.results)
            json.dump(results_json, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_dir}")
    
    def _convert_numpy_types(self, obj):
        """
        Convert numpy types to native Python types for JSON serialization.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

def main():
    """
    Example usage of MomentumStrategy.
    """
    # Initialize strategy
    strategy = MomentumStrategy()
    
    # Run full analysis
    results = strategy.run_full_analysis()
    
    # Display key results
    print("\nMomentum Strategy Analysis Results:")
    print("=" * 50)
    
    # Data summary
    data_summary = results.get('data_summary', {})
    print(f"\nData Summary:")
    print(f"  Total observations: {data_summary.get('total_observations', 'N/A'):,}")
    print(f"  Unique stocks: {data_summary.get('unique_stocks', 'N/A'):,}")
    print(f"  Date range: {data_summary.get('date_range', {}).get('start', 'N/A')} to {data_summary.get('date_range', {}).get('end', 'N/A')}")
    
    # Winner-loser spreads
    spreads = strategy.get_winner_loser_spreads()
    if not spreads.empty:
        print(f"\nWinner-Loser Spreads:")
        for _, row in spreads.iterrows():
            print(f"  ({row['formation_period']},{row['holding_period']}): {row['mean_spread']:.4f} (t={row['t_stat']:.2f})")
    
    # Save results
    strategy.save_results()
    print(f"\nResults saved to {config.RESULTS_DIR}")

if __name__ == "__main__":
    main()