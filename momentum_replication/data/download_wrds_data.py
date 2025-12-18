"""
WRDS Data Download Script for Jegadeesh & Titman (1993) Momentum Strategy Replication

This script downloads the complete CRSP dataset and market data for the J&T sample period (1965-1989) 
and saves them locally to avoid future WRDS connection issues.

Usage:
    python momentum_replication/data/download_wrds_data.py
"""

import sys
from pathlib import Path
import os

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
import config

# Import DataLoader from same directory
from data_loader import DataLoader


def check_existing_files():
    """Check if data files already exist locally."""
    data_dir = Path(__file__).parent
    stock_file = data_dir / 'stock_data_raw.csv'
    market_file = data_dir / 'market_data.csv'
    first_week_file = data_dir / 'first_week_returns.csv'
    
    files_exist = {
        'stock_data': stock_file.exists(),
        'market_data': market_file.exists(),
        'first_week_returns': first_week_file.exists()
    }
    
    return files_exist, stock_file, market_file, first_week_file


def download_wrds_data():
    """
    Download CRSP stock data and market data using the existing DataLoader class.
    Checks for existing files first and skips download if they exist.
    """
    print("🚀 WRDS DATA DOWNLOAD FOR JEGADEESH & TITMAN (1993) REPLICATION")
    print("=" * 70)
    
    # Check for existing files first
    files_exist, stock_file, market_file, first_week_file = check_existing_files()
    
    print("📁 Checking for existing data files...")
    
    if files_exist['stock_data']:
        print(f"✅ Stock data already exists: {stock_file}")
        print("   📊 Skipping stock data download")
    
    if files_exist['market_data']:
        print(f"✅ Market data already exists: {market_file}")
        print("   📈 Skipping market data download")
    
    if files_exist['first_week_returns']:
        print(f"✅ First-week returns already exists: {first_week_file}")
        print("   🗓️ Skipping first-week returns download")

    
    # If both files exist, no need to download anything
    if all(files_exist.values()):
        print(f"\n🎉 ALL DATA ALREADY AVAILABLE LOCALLY!")
        print(f"💡 Both stock, market and weekly data are saved and ready to use")
        return True
    
    # If some data is missing, proceed with download
    print(f"\n🔧 Some data files missing. Initializing WRDS connection...")
    
    try:
        loader = DataLoader(data_source='wrds')
        success = True
        
        # Download stock data if needed
        if not files_exist['stock_data']:
            print(f"\n📊 Downloading CRSP stock data from {config.START_DATE} to {config.END_DATE}...")
            stock_data = loader.load_stock_data(
                start_date=config.START_DATE,
                end_date=config.END_DATE,
                save_to_csv=True
            )
            
            if stock_data is not None and not stock_data.empty:
                print(f"✅ Stock data downloaded: {len(stock_data):,} observations")
                print(f"🏢 {stock_data['permno'].nunique():,} unique stocks")
                print(f"📅 Date range: {stock_data['date'].min().date()} to {stock_data['date'].max().date()}")
                print(f"💾 Saved to: {stock_file}")
            else:
                print("❌ Failed to download stock data")
                success = False
        
        # Download market data if needed
        if not files_exist['market_data']:
            print(f"\n📈 Downloading market and risk-free rate data...")
            market_data = loader.load_market_data()
            
            if market_data is not None and not market_data.empty:
                print(f"✅ Market data downloaded: {len(market_data):,} observations")
                print(f"📅 Date range: {market_data['date'].min().date()} to {market_data['date'].max().date()}")
                print(f"💾 Saved to: {market_file}")
            else:
                print("❌ Failed to download market data")
                success = False
        
        # Download first-week returns if needed
        if not files_exist['first_week_returns']:
            print(f"\n🗓️ Downloading first-week (5 trading days) returns from daily CRSP DSF...")
            first_week = loader.load_first_week_returns(
                start_date=config.START_DATE,
                end_date=config.END_DATE,
                save_to_csv=True
            )

            if first_week is not None and not first_week.empty:
                print(f"✅ First-week returns downloaded: {len(first_week):,} permno-month observations")
                print(f"💾 Saved to: {first_week_file}")
            else:
                print("❌ Failed to download first-week returns")
                success = False

        return success
            
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return False


def main():
    """Main function to run the download."""
    success = download_wrds_data()
    
    if success:
        print("\n🎉 DATA READY FOR MOMENTUM ANALYSIS!")
        print("💡 Both stock and market data are available locally")
        print("🚀 You can now run the Jegadeesh & Titman momentum strategy")
    else:
        print("\n❌ DATA DOWNLOAD FAILED!")
        print("💡 Check your WRDS credentials and connection")
        print("🔧 Make sure Duo authentication is working properly")


if __name__ == "__main__":
    main()