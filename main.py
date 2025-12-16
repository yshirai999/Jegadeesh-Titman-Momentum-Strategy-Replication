"""
Main execution script for Jegadeesh & Titman (1993) momentum strategy replication.
Runs the complete analysis and generates results tables matching the original paper.
"""

import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')

import config
from momentum_strategy import MomentumStrategy
from results_generator import ResultsGenerator
from plotting import MomentumPlotter
from utils import setup_logging, ensure_directory_exists

# Set up logging
logger = setup_logging(config.LOG_FILE, config.LOGGING_LEVEL)


def main():
    """
    Main function to run complete Jegadeesh & Titman (1993) momentum strategy replication.
    """
    logger.info("Starting Jegadeesh & Titman (1993) momentum strategy replication")
    logger.info("=" * 80)
    try:
        # Create output directories
        ensure_directory_exists(config.RESULTS_DIR)
        ensure_directory_exists(config.FIGURES_DIR)
        ensure_directory_exists(config.DATA_DIR)
        
        print("\n" + "=" * 80)
        print("JEGADEESH & TITMAN (1993) MOMENTUM STRATEGY REPLICATION")
        print("=" * 80)
        print(f"Sample Period: {config.START_DATE.strftime('%Y-%m-%d')} to {config.END_DATE.strftime('%Y-%m-%d')}")
        print(f"Formation Periods (J): {config.FORMATION_PERIODS} months")
        print(f"Holding Periods (K): {config.HOLDING_PERIODS} months")
        print(f"Portfolio Weighting: {config.PORTFOLIO_WEIGHTING.title()}")
        print(f"Number of Portfolios: {config.NUM_PORTFOLIOS}")
        print(f"Skip Period: {config.SKIP_PERIOD} month(s)")
        
        # Step 1: Initialize and run momentum strategy
        print("[1/5] Running momentum strategy analysis...")
        strategy = MomentumStrategy(
            formation_periods=config.FORMATION_PERIODS,
            holding_periods=config.HOLDING_PERIODS,
            num_portfolios=config.NUM_PORTFOLIOS,
            weighting_scheme=config.PORTFOLIO_WEIGHTING
        )
        
        # Run full analysis
        results = strategy.run_full_analysis(
            data_source=config.DATA_SOURCE,
            validate_data=True
        )
        
        # Display basic results
        data_summary = results.get('data_summary', {})
        print(f"  ✓ Loaded {data_summary.get('total_observations', 'N/A'):,} observations")
        print(f"  ✓ Analyzed {data_summary.get('unique_stocks', 'N/A'):,} unique stocks")
        print(f"  ✓ Calculated returns for {len(strategy.strategy_returns)} (J,K) combinations")
        
        # Step 2: Generate results tables
        print("[2/5] Generating results tables...")
        results_generator = ResultsGenerator(strategy)        
        
        # Generate main results tables
        tables = results_generator.generate_all_tables()
        
        # Save tables
        results_generator.save_tables(tables)
        print(f"  ✓ Generated {len(tables)} results tables")
        
        # Step 3: Create visualizations
        print("[3/5] Creating visualizations...")
        plotter = MomentumPlotter(strategy)
        
        # Generate plots
        plots_created = plotter.create_all_plots()
        print(f"  ✓ Created {plots_created} plots and charts")
        
        # Step 4: Display key results
        print("[4/5] Displaying key results...")
        display_key_results(strategy, tables)
        
        # Step 5: Save comprehensive results
        print("[5/5] Saving comprehensive results...")
        strategy.save_results()
        
        # Create summary report
        create_summary_report(strategy, tables)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Results saved to: {Path(config.RESULTS_DIR).absolute()}")
        print(f"Figures saved to: {Path(config.FIGURES_DIR).absolute()}")
        print(f"Log file: {config.LOG_FILE}")
        logger.info("Momentum strategy replication completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("Check the log file for detailed error information.")
        raise



def display_key_results(strategy: MomentumStrategy, tables: dict):
    """
    Display key results from the momentum strategy analysis.
    
    Parameters:
    -----------
    strategy : MomentumStrategy
        Completed momentum strategy analysis
    tables : dict
        Generated results tables
    """
    # Display winner-loser spreads
    if 'winner_loser_spreads' in tables:
        spreads_table = tables['winner_loser_spreads']
        print("\n  Winner-Loser Portfolio Spreads:")
        print("  " + "-" * 50)
        for _, row in spreads_table.iterrows():
            j, k = row['Formation'], row['Holding']
            mean_ret = row['Mean Return (%)']
            t_stat = row['t-statistic']
            significance = "***" if abs(t_stat) > 2.576 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.645 else ""
            print(f"  ({j:2d},{k:2d}): {mean_ret:6.2f}% (t={t_stat:5.2f}){significance}")
    
    # Display portfolio performance summary
    if hasattr(strategy, 'strategy_returns') and strategy.strategy_returns:
        print("\n  Portfolio Performance Summary:")
        print("  " + "-" * 50)
        for (j, k), returns_data in list(strategy.strategy_returns.items())[:3]:  
            # Show first 3
            if not returns_data.empty:
                performance = strategy.returns_calculator.calculate_performance_metrics(returns_data)
                if 'long_short' in performance:
                    ls_metrics = performance['long_short']
                    ann_ret = ls_metrics.get('annualized_return', np.nan) * 100
                    ann_vol = ls_metrics.get('annualized_volatility', np.nan) * 100
                    sharpe = ls_metrics.get('sharpe_ratio', np.nan)
                    print(f"  ({j},{k}) Long-Short: {ann_ret:6.2f}% return, {ann_vol:5.2f}% vol, {sharpe:5.2f} Sharpe")
    
    # Display data quality summary
    if hasattr(strategy, 'data_validator') and strategy.data_validator:
        validation_results = strategy.data_validator.validation_results
        quality_score = validation_results.get('data_quality_score', 0)
        warnings_count = len(validation_results.get('warnings', []))
        errors_count = len(validation_results.get('errors', []))
        print(f"\n  Data Quality Summary:")
        print(f"  " + "-" * 50)
        print(f"  Quality Score: {quality_score:.1f}/100")
        print(f"  Warnings: {warnings_count}, Errors: {errors_count}")


def create_summary_report(strategy: MomentumStrategy, tables: dict):
    """
    Create a comprehensive summary report.
    
    Parameters:
    -----------
    strategy : MomentumStrategy
        Completed momentum strategy analysis
    tables : dict
        Generated results tables
    """
    report_path = Path(config.RESULTS_DIR) / "summary_report.txt"
    with open(report_path, 'w') as f:
        f.write("JEGADEESH & TITMAN (1993) MOMENTUM STRATEGY REPLICATION\n")
        f.write("=" * 65 + "\n\n")
        
        # Analysis parameters
        f.write("ANALYSIS PARAMETERS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Sample Period: {config.START_DATE} to {config.END_DATE}\n")
        f.write(f"Formation Periods: {config.FORMATION_PERIODS} months\n")
        f.write(f"Holding Periods: {config.HOLDING_PERIODS} months\n")
        f.write(f"Portfolio Weighting: {config.PORTFOLIO_WEIGHTING}\n")
        f.write(f"Number of Portfolios: {config.NUM_PORTFOLIOS}\n")
        f.write(f"Skip Period: {config.SKIP_PERIOD} month(s)\n")
        f.write(f"Data Source: {config.DATA_SOURCE}\n\n")
        
        # Data summary
        data_summary = strategy.results.get('data_summary', {})
        f.write("DATA SUMMARY\n")
        f.write("-" * 12 + "\n")
        f.write(f"Total Observations: {data_summary.get('total_observations', 'N/A'):,}\n")
        f.write(f"Unique Stocks: {data_summary.get('unique_stocks', 'N/A'):,}\n")
        f.write(f"Monthly Observations: {data_summary.get('monthly_observations', 'N/A'):,}\n")
        f.write(f"Average Stocks per Month: {data_summary.get('avg_stocks_per_month', 0):.1f}\n\n")
        
        # Key findings
        f.write("KEY FINDINGS\n")
        f.write("-" * 12 + "\n")
        
        if 'winner_loser_spreads' in tables:
            spreads_table = tables['winner_loser_spreads']
            
            # Find best performing strategy
            best_strategy = spreads_table.loc[spreads_table['Mean Return (%)'].idxmax()]
            f.write(f"Best Strategy: ({best_strategy['Formation']},{best_strategy['Holding']}) ")
            f.write(f"with {best_strategy['Mean Return (%)']:.2f}% monthly return\n")
            
            # Count significant strategies
            significant_strategies = (spreads_table['t-statistic'].abs() > 1.96).sum()
            f.write(f"Statistically Significant Strategies (5% level): {significant_strategies}/{len(spreads_table)}\n")
            
            # Average momentum profit
            avg_momentum = spreads_table['Mean Return (%)'].mean()
            f.write(f"Average Momentum Profit: {avg_momentum:.2f}% per month\n\n")
        
        # Generated files
        f.write("GENERATED FILES\n")
        f.write("-" * 15 + "\n")
        
        results_dir = Path(config.RESULTS_DIR)
        figures_dir = Path(config.FIGURES_DIR)
        
        # List CSV files
        csv_files = list(results_dir.glob("*.csv"))
        f.write(f"Results Tables ({len(csv_files)} files):\n")
        for file in sorted(csv_files):
            f.write(f"  - {file.name}\n")
        
        # List figure files
        fig_files = list(figures_dir.glob("*.png")) + list(figures_dir.glob("*.pdf"))
        f.write(f"\nFigures ({len(fig_files)} files):\n")
        for file in sorted(fig_files):
            f.write(f"  - {file.name}\n")
        
        f.write(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"  ✓ Summary report saved to: {report_path}")
    
    if __name__ == "__main__":
        main()