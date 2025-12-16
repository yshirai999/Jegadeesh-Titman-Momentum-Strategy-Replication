"""
Date handling utilities for Jegadeesh & Titman (1993) momentum strategy replication.
Provides specialized date functions for financial data processing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Tuple, Union, Optional
import calendar

import config

def get_month_end_dates(start_date: Union[str, datetime, pd.Timestamp], 
                       end_date: Union[str, datetime, pd.Timestamp]) -> List[pd.Timestamp]:
    """
    Get list of month-end dates between start and end dates.
    
    Parameters:
    -----------
    start_date : Union[str, datetime, pd.Timestamp]
        Start date
    end_date : Union[str, datetime, pd.Timestamp]
        End date
        
    Returns:
    --------
    List[pd.Timestamp]
        List of month-end dates
    """
    # Convert to pandas timestamps
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    
    # Generate monthly periods
    month_ends = pd.date_range(start=start_ts, end=end_ts, freq='M')
    
    return month_ends.tolist()

def add_months(date: Union[str, datetime, pd.Timestamp], 
               months: int) -> pd.Timestamp:
    """
    Add months to a date, handling month-end dates properly.
    
    Parameters:
    -----------
    date : Union[str, datetime, pd.Timestamp]
        Starting date
    months : int
        Number of months to add (can be negative)
        
    Returns:
    --------
    pd.Timestamp
        Date with months added
    """
    date_ts = pd.to_datetime(date)
    return date_ts + pd.DateOffset(months=months)

def subtract_months(date: Union[str, datetime, pd.Timestamp], 
                   months: int) -> pd.Timestamp:
    """
    Subtract months from a date.
    
    Parameters:
    -----------
    date : Union[str, datetime, pd.Timestamp]
        Starting date
    months : int
        Number of months to subtract
        
    Returns:
    --------
    pd.Timestamp
        Date with months subtracted
    """
    return add_months(date, -months)

def get_formation_period_start(current_date: Union[str, datetime, pd.Timestamp], 
                              formation_months: int,
                              skip_months: int = 1) -> pd.Timestamp:
    """
    Get start date for formation period given current date.
    
    Parameters:
    -----------
    current_date : Union[str, datetime, pd.Timestamp]
        Current portfolio formation date
    formation_months : int
        Number of months in formation period
    skip_months : int
        Number of months to skip (default 1 to avoid microstructure issues)
        
    Returns:
    --------
    pd.Timestamp
        Start date for formation period
    """
    current_ts = pd.to_datetime(current_date)
    
    # Skip months and then go back formation_months
    formation_end = subtract_months(current_ts, skip_months)
    formation_start = subtract_months(formation_end, formation_months - 1)
    
    return formation_start

def get_formation_period_end(current_date: Union[str, datetime, pd.Timestamp], 
                            skip_months: int = 1) -> pd.Timestamp:
    """
    Get end date for formation period given current date.
    
    Parameters:
    -----------
    current_date : Union[str, datetime, pd.Timestamp]
        Current portfolio formation date
    skip_months : int
        Number of months to skip
        
    Returns:
    --------
    pd.Timestamp
        End date for formation period
    """
    current_ts = pd.to_datetime(current_date)
    return subtract_months(current_ts, skip_months)

def get_holding_period_dates(formation_date: Union[str, datetime, pd.Timestamp],
                           holding_months: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get start and end dates for holding period.
    
    Parameters:
    -----------
    formation_date : Union[str, datetime, pd.Timestamp]
        Portfolio formation date
    holding_months : int
        Number of months to hold portfolio
        
    Returns:
    --------
    Tuple[pd.Timestamp, pd.Timestamp]
        Start and end dates for holding period
    """
    formation_ts = pd.to_datetime(formation_date)
    
    # Holding period starts after formation date
    holding_start = add_months(formation_ts, 1)
    holding_end = add_months(holding_start, holding_months - 1)
    
    return holding_start, holding_end

def main():
    """
    Example usage of date utilities.
    """
    print("Date Utilities Example")
    print("=" * 30)
    
    # Example sample period
    start_date = config.START_DATE
    end_date = config.END_DATE
    
    # Get month-end dates
    month_ends = get_month_end_dates(start_date, end_date)
    print(f"Sample period: {start_date} to {end_date}")
    print(f"Number of month-ends: {len(month_ends)}")
    
    # Example of formation and holding periods
    formation_date = month_ends[12]  # Use 12th month-end as example
    formation_start = get_formation_period_start(formation_date, 6, 1)
    formation_end = get_formation_period_end(formation_date, 1)
    holding_start, holding_end = get_holding_period_dates(formation_date, 6)
    
    print(f"\\nExample for formation date {formation_date.strftime('%Y-%m')}:")
    print(f"Formation period: {formation_start.strftime('%Y-%m')} to {formation_end.strftime('%Y-%m')}")
    print(f"Holding period: {holding_start.strftime('%Y-%m')} to {holding_end.strftime('%Y-%m')}")

if __name__ == "__main__":
    main()
        -----------
        start_date : Union[str, datetime]
            Start date
        end_date : Union[str, datetime]
            End date
            
        Returns:
        --------
        List[pd.Timestamp]
            List of month-start dates
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        return pd.date_range(start=start, end=end, freq='MS').tolist()
    
    @staticmethod
    def add_months(date: Union[str, datetime, pd.Timestamp], 
                   months: int) -> pd.Timestamp:
        """
        Add or subtract months from a date.
        
        Parameters:
        -----------
        date : Union[str, datetime, pd.Timestamp]
            Starting date
        months : int
            Number of months to add (negative to subtract)
            
        Returns:
        --------
        pd.Timestamp
            New date
        """
        date_ts = pd.to_datetime(date)
        return date_ts + pd.DateOffset(months=months)
    
    @staticmethod
    def get_last_day_of_month(year: int, month: int) -> pd.Timestamp:
        """
        Get the last day of a specific month and year.
        
        Parameters:
        -----------
        year : int
            Year
        month : int
            Month (1-12)
            
        Returns:
        --------
        pd.Timestamp
            Last day of the month
        """
        last_day = calendar.monthrange(year, month)[1]
        return pd.Timestamp(year, month, last_day)
    
    @staticmethod
    def align_to_month_end(date: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
        """
        Align date to the end of its month.
        
        Parameters:
        -----------
        date : Union[str, datetime, pd.Timestamp]
            Input date
            
        Returns:
        --------
        pd.Timestamp
            Month-end date
        """
        date_ts = pd.to_datetime(date)
        return DateUtils.get_last_day_of_month(date_ts.year, date_ts.month)
    
    @staticmethod
    def get_formation_period_dates(end_date: Union[str, datetime, pd.Timestamp],
                                  formation_months: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get start and end dates for a formation period.
        
        Parameters:
        -----------
        end_date : Union[str, datetime, pd.Timestamp]
            Formation period end date
        formation_months : int
            Length of formation period in months
            
        Returns:
        --------
        Tuple[pd.Timestamp, pd.Timestamp]
            (start_date, end_date) of formation period
        """
        end_ts = pd.to_datetime(end_date)
        start_ts = DateUtils.add_months(end_ts, -formation_months + 1)
        
        return start_ts, end_ts
    
    @staticmethod
    def get_holding_period_dates(start_date: Union[str, datetime, pd.Timestamp],
                               holding_months: int,
                               skip_months: int = 0) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get start and end dates for a holding period.
        
        Parameters:
        -----------
        start_date : Union[str, datetime, pd.Timestamp]
            Base start date (typically formation end date)
        holding_months : int
            Length of holding period in months
        skip_months : int
            Number of months to skip before holding period starts
            
        Returns:
        --------
        Tuple[pd.Timestamp, pd.Timestamp]
            (start_date, end_date) of holding period
        """
        base_date = pd.to_datetime(start_date)
        holding_start = DateUtils.add_months(base_date, skip_months)
        holding_end = DateUtils.add_months(holding_start, holding_months - 1)
        
        return holding_start, holding_end
    
    @staticmethod
    def get_overlapping_portfolio_dates(formation_date: Union[str, datetime, pd.Timestamp],
                                      formation_months: int,
                                      holding_months: int,
                                      skip_months: int = 0) -> dict:
        """
        Get all relevant dates for overlapping portfolio strategy.
        
        Parameters:
        -----------
        formation_date : Union[str, datetime, pd.Timestamp]
            Portfolio formation date
        formation_months : int
            Formation period length
        holding_months : int
            Holding period length
        skip_months : int
            Skip period length
            
        Returns:
        --------
        dict
            Dictionary with formation_start, formation_end, holding_start, holding_end
        """
        formation_end = pd.to_datetime(formation_date)
        formation_start = DateUtils.add_months(formation_end, -formation_months + 1)
        
        holding_start = DateUtils.add_months(formation_end, skip_months)
        holding_end = DateUtils.add_months(holding_start, holding_months - 1)
        
        return {
            'formation_start': formation_start,
            'formation_end': formation_end,
            'holding_start': holding_start,
            'holding_end': holding_end
        }
    
    @staticmethod
    def generate_rebalance_dates(start_date: Union[str, datetime, pd.Timestamp],
                               end_date: Union[str, datetime, pd.Timestamp],
                               frequency: str = 'monthly') -> List[pd.Timestamp]:
        """
        Generate portfolio rebalancing dates.
        
        Parameters:
        -----------
        start_date : Union[str, datetime, pd.Timestamp]
            Start date
        end_date : Union[str, datetime, pd.Timestamp]
            End date
        frequency : str
            Rebalancing frequency ('monthly', 'quarterly', 'annually')
            
        Returns:
        --------
        List[pd.Timestamp]
            List of rebalancing dates
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if frequency == 'monthly':
            freq = 'M'
        elif frequency == 'quarterly':
            freq = 'Q'
        elif frequency == 'annually':
            freq = 'A'
        else:
            raise ValueError(f"Unknown frequency: {frequency}")
        
        return pd.date_range(start=start, end=end, freq=freq).tolist()
    
    @staticmethod
    def is_month_end(date: Union[str, datetime, pd.Timestamp]) -> bool:
        """
        Check if date is the last day of the month.
        
        Parameters:
        -----------
        date : Union[str, datetime, pd.Timestamp]
            Date to check
            
        Returns:
        --------
        bool
            True if date is month-end
        """
        date_ts = pd.to_datetime(date)
        month_end = DateUtils.get_last_day_of_month(date_ts.year, date_ts.month)
        return date_ts.date() == month_end.date()
    
    @staticmethod
    def get_business_days_in_month(year: int, month: int) -> int:
        """
        Get number of business days in a month.
        
        Parameters:
        -----------
        year : int
            Year
        month : int
            Month
            
        Returns:
        --------
        int
            Number of business days
        """
        start_date = pd.Timestamp(year, month, 1)
        end_date = DateUtils.get_last_day_of_month(year, month)
        
        business_days = pd.bdate_range(start=start_date, end=end_date)
        return len(business_days)
    
    @staticmethod
    def get_years_between_dates(start_date: Union[str, datetime, pd.Timestamp],
                              end_date: Union[str, datetime, pd.Timestamp]) -> float:
        """
        Calculate number of years between two dates.
        
        Parameters:
        -----------
        start_date : Union[str, datetime, pd.Timestamp]
            Start date
        end_date : Union[str, datetime, pd.Timestamp]
            End date
            
        Returns:
        --------
        float
            Number of years (fractional)
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        days_diff = (end - start).days
        return days_diff / 365.25  # Account for leap years
    
    @staticmethod
    def create_date_index(data: pd.DataFrame, 
                         date_column: str = 'date',
                         sort: bool = True) -> pd.DataFrame:
        """
        Create and set date index for DataFrame.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input DataFrame
        date_column : str
            Name of date column
        sort : bool
            Whether to sort by date
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with date index
        """
        df = data.copy()
        
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found")
        
        # Convert to datetime if not already
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Set as index
        df = df.set_index(date_column)
        
        # Sort by date if requested
        if sort:
            df = df.sort_index()
        
        return df
    
    @staticmethod
    def align_time_series(data1: pd.DataFrame,
                         data2: pd.DataFrame,
                         date_column1: str = 'date',
                         date_column2: str = 'date',
                         how: str = 'inner') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align two time series datasets by their date columns.
        
        Parameters:
        -----------
        data1 : pd.DataFrame
            First dataset
        data2 : pd.DataFrame
            Second dataset
        date_column1 : str
            Date column name in first dataset
        date_column2 : str
            Date column name in second dataset
        how : str
            How to align ('inner', 'outer', 'left', 'right')
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Aligned datasets
        """
        # Get unique dates from both datasets
        dates1 = set(pd.to_datetime(data1[date_column1]))
        dates2 = set(pd.to_datetime(data2[date_column2]))
        
        # Determine common dates based on alignment method
        if how == 'inner':
            common_dates = dates1 & dates2
        elif how == 'outer':
            common_dates = dates1 | dates2
        elif how == 'left':
            common_dates = dates1
        elif how == 'right':
            common_dates = dates2
        else:
            raise ValueError(f"Unknown alignment method: {how}")
        
        # Filter datasets to common dates
        aligned_data1 = data1[data1[date_column1].isin(common_dates)].copy()
        aligned_data2 = data2[data2[date_column2].isin(common_dates)].copy()
        
        return aligned_data1, aligned_data2
    
    @staticmethod
    def get_sample_period_dates() -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get sample period start and end dates from config.
        
        Returns:
        --------
        Tuple[pd.Timestamp, pd.Timestamp]
            (start_date, end_date) from configuration
        """
        return pd.to_datetime(config.START_DATE), pd.to_datetime(config.END_DATE)
    
    @staticmethod
    def get_extended_sample_dates() -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get extended sample period dates (including pre-sample for momentum calculation).
        
        Returns:
        --------
        Tuple[pd.Timestamp, pd.Timestamp]
            (extended_start_date, end_date) from configuration
        """
        return pd.to_datetime(config.EXTENDED_START_DATE), pd.to_datetime(config.END_DATE)
    
    @staticmethod
    def validate_sample_coverage(data: pd.DataFrame,
                               date_column: str = 'date',
                               min_coverage: float = 0.8) -> bool:
        """
        Validate that data covers sufficient portion of sample period.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to validate
        date_column : str
            Date column name
        min_coverage : float
            Minimum required coverage (0.0 to 1.0)
            
        Returns:
        --------
        bool
            True if coverage is sufficient
        """
        sample_start, sample_end = DateUtils.get_sample_period_dates()
        
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found")
        
        data_dates = pd.to_datetime(data[date_column])
        data_start = data_dates.min()
        data_end = data_dates.max()
        
        # Calculate coverage
        sample_months = DateUtils.get_month_end_dates(sample_start, sample_end)
        data_months = DateUtils.get_month_end_dates(
            max(data_start, sample_start),
            min(data_end, sample_end)
        )
        
        coverage = len(data_months) / len(sample_months)
        
        if coverage < min_coverage:
            warnings.warn(f"Low sample coverage: {coverage:.1%} < {min_coverage:.1%}")
            return False
        
        return True
    
    @staticmethod
    def create_momentum_calendar(j_months: int, k_months: int) -> pd.DataFrame:
        """
        Create a calendar showing formation and holding periods for momentum strategy.
        
        Parameters:
        -----------
        j_months : int
            Formation period length
        k_months : int
            Holding period length
            
        Returns:
        --------
        pd.DataFrame
            Calendar with formation and holding period information
        """
        sample_start, sample_end = DateUtils.get_sample_period_dates()
        
        # Get all possible formation end dates
        formation_dates = DateUtils.get_month_end_dates(
            DateUtils.add_months(sample_start, j_months - 1),
            sample_end
        )
        
        calendar_data = []
        
        for formation_end in formation_dates:
            dates_info = DateUtils.get_overlapping_portfolio_dates(
                formation_end, j_months, k_months, config.SKIP_PERIOD
            )
            
            calendar_data.append({
                'formation_end': formation_end,
                'formation_start': dates_info['formation_start'],
                'holding_start': dates_info['holding_start'],
                'holding_end': dates_info['holding_end'],
                'formation_months': j_months,
                'holding_months': k_months,
                'skip_months': config.SKIP_PERIOD
            })
        
        return pd.DataFrame(calendar_data)

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_monthly_dates(start: Union[str, datetime], 
                     end: Union[str, datetime]) -> List[pd.Timestamp]:
    """Convenience function to get month-end dates."""
    return DateUtils.get_month_end_dates(start, end)

def add_months_to_date(date: Union[str, datetime], months: int) -> pd.Timestamp:
    """Convenience function to add months to a date."""
    return DateUtils.add_months(date, months)

def get_jt_sample_dates() -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get the original Jegadeesh & Titman sample period dates."""
    return DateUtils.get_sample_period_dates()

def create_trading_calendar() -> pd.DataFrame:
    """Create a comprehensive trading calendar for the sample period."""
    start_date, end_date = DateUtils.get_extended_sample_dates()
    
    # Get all month-end dates
    month_ends = DateUtils.get_month_end_dates(start_date, end_date)
    
    calendar_data = []
    for date in month_ends:
        calendar_data.append({
            'date': date,
            'year': date.year,
            'month': date.month,
            'quarter': date.quarter,
            'is_month_end': True,
            'business_days_in_month': DateUtils.get_business_days_in_month(date.year, date.month)
        })
    
    return pd.DataFrame(calendar_data)