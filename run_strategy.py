"""
Run Momentum Strategy Script
============================

Convenience script to run the Jegadeesh & Titman (1993) momentum strategy
from the project root directory.

Usage:
    python run_strategy.py

This will run all 32 strategy combinations and save results to the results/ folder.
"""

import sys
import os

# Add momentum_replication to Python path
momentum_path = os.path.join(os.path.dirname(__file__), 'momentum_replication')
sys.path.insert(0, momentum_path)

# Import the momentum strategy module directly
import importlib.util
strategy_path = os.path.join(momentum_path, 'strategies', 'momentum_strategy.py')
spec = importlib.util.spec_from_file_location("momentum_strategy", strategy_path)
momentum_strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(momentum_strategy)

if __name__ == "__main__":
    momentum_strategy.main()