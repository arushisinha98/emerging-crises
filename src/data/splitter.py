import pandas as pd
import numpy as np
from typing import Tuple, Set, Union, Optional

from .data_utilities import build_labels

class DataSplitter:
    """
    A utility class for splitting time-series data into train, and test sets.

    This class provides functionality to split financial time-series data based on
    chronological order, ensuring that training data comes before test data.
    This maintains temporal consistency for time-series modeling and prevents data leakage.
    
    The splitting is performed based on specified proportions within a date range.
    If start_date or end_date are not provided, they default to the earliest and
    latest dates in the dataset respectively.
    
    Attributes:
    -----------
    start_date (pd.Timestamp or None)
        The start date for the dataset splitting.
    end_date (pd.Timestamp or None)
        The end date for the dataset splitting.
    train_prop (float)
        Proportion of data to use for training (0.0 to 1.0).
    test_prop (float)
        Proportion of data to use for testing (0.0 to 1.0).
        
    Examples:
        >>> # Use entire dataset date range
        >>> splitter = DataSplitter(train_prop=0.8, test_prop=0.2)
        >>> 
        >>> # Specify custom date range
        >>> splitter = DataSplitter(
        ...     start_date='2020-01-01',
        ...     end_date='2023-12-31',
        ...     train_prop=0.8,
        ...     test_prop=0.2
        ... )
        >>> train_data = splitter.split(df, 'train')
        >>> test_data = splitter.split(df, 'test')
    """
    
    def __init__(self,
                 geographic: bool = False,
                 start_date: Optional[Union[str, pd.Timestamp]] = None,
                 end_date: Optional[Union[str, pd.Timestamp]] = None,
                 cutoff_date: Optional[Union[str, pd.Timestamp]] = None,
                 train_prop: float = 0.8,
                 test_prop: float = 0.2) -> None:
        """
        Initialize the DataSplitter with specified proportions and date range.
        
        Parameters:
        -----------
        start_date: str, pd.Timestamp
            The start date for splitting the dataset.
            If None, uses the earliest date in the dataset.
        end_date: str, pd.Timestamp
            The end date for splitting the dataset.
            If None, uses the latest date in the dataset.
        cutoff_date: str, pd.Timestamp
            The cutoff data for splitting the dataset.
            If None, performs an optimized temporal split.
        train_prop: float
            Proportion of data to allocate for training.
            Must be between 0.0 and 1.0. Default is 0.8.
        test_prop: float
            Proportion of data to allocate for testing.
            Must be between 0.0 and 1.0. Default is 0.1.
                
        Raises:
        -------
        ValueError: If proportions don't sum to 1.0 or are outside valid range.
        TypeError: If start_date or end_date cannot be converted to datetime.

        Note:
        -----
            If cutoff_date is provided, it must be within the range of start_date and end_date.
            If both start_date and end_date are None, they will be determined from the data.
            The sum of train_prop and test_prop should equal 1.0.

        """
        # Validate proportions
        if not all(0.0 <= prop <= 1.0 for prop in [train_prop, test_prop]):
            raise ValueError("All proportions must be between 0.0 and 1.0")
            
        if abs(train_prop + test_prop - 1.0) > 1e-6:
            raise ValueError(f"Proportions must sum to 1.0, got {train_prop + test_prop}")
        
        # Validate and convert dates (if provided)
        try:
            self.start_date = pd.to_datetime(start_date) if start_date is not None else None
        except (ValueError, TypeError) as e:
            raise TypeError(f"start_date must be convertible to datetime: {e}")
            
        try:
            self.end_date = pd.to_datetime(end_date) if end_date is not None else None
        except (ValueError, TypeError) as e:
            raise TypeError(f"end_date must be convertible to datetime: {e}")
        
        self.cutoff_date = None
        if cutoff_date is not None:
            try:
                self.cutoff_date = pd.to_datetime(cutoff_date)
                self.geographic = False
            except (ValueError, TypeError) as e:
                raise TypeError(f"cutoff_date must be convertible to datetime: {e}")
        
        # Temporal split is better than geographic split due to data leakage
        if self.cutoff_date is None:
            self.geographic = geographic

        self.train_prop = train_prop
        self.test_prop = test_prop

        self.train_df = None
        self.test_df  = None

    def split_type(self):
        """
        Get the type of split being performed.
        
        Returns:
        --------
        str
            'geographic' if splitting by geography, 'temporal' if by time.
        """
        return "geographic" if self.geographic else "temporal"

    def perform_split(self, df: pd.DataFrame,
                      beta: float = 0.8) -> None:
        """
        Split the DataFrame into train or test set based on
        (1) custom date, or
        (2) temporal order, 
        (3) geographic grouping
        
        If cutoff date was provided, this method splits the data into train and test sets
        chronologically, such that:
        - Training data contains the earliest time periods
        - Test data contains the most recent time periods

        If self.geographic is True, this method splits the data by geography.

        If self.geographic is False and no date is provided, this method splits the
        data chronologically. Due to imbalance in class labels, train_prop and test_prop
        are used as a guide to determine the proportion of minority class labels 
        allocated to each split. Parameter beta is used as a weighting factor for label
        proportion deviation (default is 0.8).
        
        Parameters:
        -----------
        df: pd.DataFrame
            The input DataFrame to split.
        beta: float
            Weighting factor for label proportion deviation
           
        Raises:
        -------
        TypeError: If date information cannot be converted to datetime.
        
        """
        df_copy = df.reset_index(drop=True)
        try:
            dates = pd.to_datetime(df_copy['Date'])
        except Exception as e:
            raise TypeError(f"'Date' must be datetime‐convertible: {e}")

        # Filter data to specified date range
        actual_start = self.start_date or dates.min()
        actual_end = self.end_date or dates.max()
        mask = (dates >= actual_start) & (dates <= actual_end)
        df_filtered = df_copy.loc[mask].reset_index(drop=True)
        dates_filtered = dates[mask]

        # If date is provided, perform custom split
        if self.cutoff_date is not None:
            try:
                split_date = pd.to_datetime(self.cutoff_date)
            except (ValueError, TypeError) as e:
                raise TypeError(f"date must be convertible to datetime: {e}")
            
            # Update self.train_df and self.test_df
            mask1 = dates_filtered.dt.to_period('D') <= split_date.to_period('D')
            self.train_df = df_filtered.loc[mask1].reset_index(drop=True)
            mask2 = dates_filtered.dt.to_period('D') > split_date.to_period('D')
            self.test_df = df_filtered.loc[mask2].reset_index(drop=True)

        # If geographic, perform split based on country and label proportions
        elif self.geographic:
            counts = df_filtered['Country'].value_counts()
            total = set(counts.index)

            train_target = len(df_filtered)*self.train_prop
            train_set, _ = _find_closest_sum(train_target, counts, iterations=8, tolerance=0.01)
            remaining = counts.loc[sorted(total - train_set)]

            test_target = len(df_filtered)*self.test_prop
            test_set, _ = _find_closest_sum(test_target, remaining, iterations=8, tolerance=0.01)

            assert train_set.isdisjoint(test_set)
            # Update self.train_df and self.test_df
            self.train_df = df_filtered[df_filtered['Country'].isin(train_set)]
            self.test_df = df_filtered[df_filtered['Country'].isin(test_set)]

        # Else, perform temporal split based on date and label proportions
        else:
            y = build_labels(df_filtered)
            
            if len(set(y)) < 2:
                raise ValueError("Cannot generate split with <2 unique labels.") 
            
            # Count unique dates by class
            unique_dates0 = sorted(dates_filtered[y == 0].dt.to_period('D').unique())
            unique_dates1 = sorted(dates_filtered[y == 1].dt.to_period('D').unique())
            
            n = min(len(unique_dates0), len(unique_dates1))
            if n == 0:
                return df_filtered.iloc[0:0], np.array([],dtype=int)
            
            # Compute cut‐point and update self.train_df and self.test_df
            optimal_cutoff = _balance_label_prop(dates_filtered, y, self.train_prop, beta)
            mask1 = dates_filtered.dt.to_period('D') <= optimal_cutoff
            self.train_df = df_filtered.loc[mask1].reset_index(drop=True)
            mask2 = dates_filtered.dt.to_period('D') > optimal_cutoff
            self.test_df = df_filtered.loc[mask2].reset_index(drop=True)
        
    def split(self, split: str, df: pd.DataFrame = None, beta: float = 0.8) -> pd.DataFrame:
        """
        Return the train or test split. If df is provided, perform split on new data using existing cutoff.
        """
        if split not in ['train', 'test']:
            raise ValueError("split must be either 'train' or 'test'")
    
        # If no splits exist yet, need df to perform initial split
        if self.train_df is None or self.test_df is None:
            if df is None:
                raise ValueError("df must be provided for initial split")
            self.perform_split(df=df, beta=beta)
        
        # If df is provided and splits already exist, apply same cutoff to new data
        elif df is not None and not df.empty:
            # Store current cutoff date for reuse
            stored_cutoff = self.cutoff_date or self.get_split_date()
            original_cutoff = self.cutoff_date
            self.cutoff_date = stored_cutoff
            self.perform_split(df=df, beta=beta)
            self.cutoff_date = original_cutoff
        
        # Return the requested split
        if split == "train":
            return self.train_df, build_labels(self.train_df)
        else:
            return self.test_df, build_labels(self.test_df)
        
    def get_split_date(self):
        if self.train_df is None or self.test_df is None:
            raise ValueError("Data has not been split yet. Call perform_split() first.")
        else:
            return self.train_df['Date'].max()

    def get_split_info(self, df: pd.DataFrame, beta: float = 0.8) -> dict:
        """
        Get information about how the data would be split without actually splitting it.
        
        This method provides useful statistics about the proposed data split,
        including date ranges and row counts for each split.
        
        Parameters:
        -----------
        df: pd.DataFrame
            The input DataFrame to analyze.
        beta: float
            Weighting factor for label proportion deviation (default is 0.8).

        Returns:
        --------
        dict: A dictionary containing split information with keys:
            - 'train': dict with 'start', 'end', 'count'
            - 'test': dict with 'start', 'end', 'count'
            - 'total_periods': int, total unique time periods
            - 'proportions': dict with actual proportions achieved
                
        Examples:
            >>> info = splitter.get_split_info(df)
            >>> print(f"Training period: {info['train']['start']} to {info['train']['end']}")
            >>> print(f"Training samples: {info['train']['count']}")
        """
        train_df, y_train = self.split(df=df, split='train', beta=beta)
        test_df, y_test = self.split(df=df, split='test', beta=beta)
        
        def get_date_range(split_df):
            if len(split_df) == 0:
                return None, None
            # Get dates from 'Date' column
            dates = pd.to_datetime(split_df['Date'])
            return dates.min(), dates.max()
        
        train_start, train_end = get_date_range(train_df)
        test_start, test_end = get_date_range(test_df)
        
        # Calculate total using filtered data (sum of all splits)
        total_count = len(train_df) + len(test_df)
        train_count = len(train_df)
        test_count = len(test_df)
        
        # Calculate actual proportions achieved
        actual_proportions = {
            'train': train_count / total_count if total_count > 0 else 0,
            'test': test_count / total_count if total_count > 0 else 0
        }

        # Calculate label proportions
        label_counts = {
            'train': (y_train == 1).mean(),
            'test': (y_test == 1).mean()
        }
        
        return {
            'train': {'start': train_start, 'end': train_end, 'count': train_count},
            'test': {'start': test_start, 'end': test_end, 'count': test_count},
            'total_periods': total_count,
            'proportions': actual_proportions,
            'label_proportions': label_counts
        }


def _find_closest_sum(target_sum: float, country_counts: pd.Series, 
                      iterations: int = 10, tolerance: float = 0.01) -> Tuple[Set[str], float]:
    """
    Algorithm to find country subset with sum closest to target.
    
    Parameters:
    -----------
    target_sum: float
        Target sum to approximate
    country_counts: pd.Series
        Series mapping countries to their data counts
    iterations: int
        Number of random restarts for better exploration
    tolerance: float
        Relative tolerance for early stopping (e.g., 0.01 = 1%)

    Returns:
    --------
    Tuple
        (best_country_set, difference_from_target)
    """
    if target_sum <= 0 or country_counts.empty:
        return set(), 0
    
    best_set = set()
    best_diff = float('inf')
    
    countries = list(country_counts.index)
    counts_dict = country_counts.to_dict()
    
    for iteration in range(iterations):
        # Initialize with different random seed for each iteration
        rng = np.random.default_rng(seed=1000 * iteration)
        shuffled_countries = rng.permutation(countries)
        
        # Greedy initialization: add countries until exceeding target
        current_sum = 0
        current_set = set()
        
        for country in shuffled_countries:
            current_sum += counts_dict[country]
            current_set.add(country)
            if current_sum >= target_sum:
                break
        
        current_diff = abs(current_sum - target_sum)
        
        if current_diff <= target_sum * tolerance:
            return current_set, current_sum - target_sum
        
        improved = True
        max_improvements = len(countries) * 2  # Prevent infinite loops
        improvement_count = 0
        
        while improved and improvement_count < max_improvements:
            improved = False
            improvement_count += 1
            
            # Try single swaps
            best_swap = None
            best_swap_diff = current_diff
            
            for remove_country in list(current_set):
                for add_country in countries:
                    if add_country not in current_set:
                        # Calculate new sum and difference
                        new_sum = current_sum - counts_dict[remove_country] + counts_dict[add_country]
                        new_diff = abs(new_sum - target_sum)
                        
                        if new_diff < best_swap_diff:
                            best_swap = (remove_country, add_country)
                            best_swap_diff = new_diff
            
            # Apply best swap if it improves the solution
            if best_swap and best_swap_diff < current_diff:
                remove_country, add_country = best_swap
                current_set.remove(remove_country)
                current_set.add(add_country)
                current_sum = current_sum - counts_dict[remove_country] + counts_dict[add_country]
                current_diff = best_swap_diff
                improved = True
        
        # Update global best if this iteration found a better solution
        if current_diff < abs(best_diff):
            best_set = current_set.copy()
            best_diff = current_sum - target_sum
        
        # Early termination across all iterations
        if abs(best_diff) <= target_sum * tolerance:
            break
    
    return best_set, best_diff

def _balance_label_prop(dates, y, train_prop, beta=0.8):
    """
    Algorithm to find the best cutoff date for balancing label proportions.

    This function iterates through candidate cutoff dates to find the one
    that minimizes the absolute difference in label proportions between
    the training and test sets, while ensuring that both sets have sufficient
    minority class samples.

    Parameters:
    -----------
    dates: pd.Series
        Series of dates corresponding to the DataFrame.
    y: pd.Series
        Series of labels (0 or 1) corresponding to the DataFrame.
    train_prop: float
        Proportion of data to allocate for training (0.0 to 1.0).
    beta: float
        Weighting factor for the label proportion deviation (default is 0.8).
    """
    unique_dates = sorted(dates.dt.to_period('D').unique())
    start_idx = len(unique_dates) // 2 # use at least half the data for training
    end_idx = int(len(unique_dates)*0.95) # leave at least 5% for testing
    candidate_cutoffs = unique_dates[start_idx:end_idx+1]
    best_cutoff = candidate_cutoffs[0]

    overall_prop = (y == 1).mean()
    best_score = 1

    for cutoff_date in candidate_cutoffs:
        # Calculate train/test splits for this cutoff
        train_mask = dates.dt.to_period('D') <= cutoff_date
        test_mask = ~train_mask

        if train_mask.sum() < 0.05*len(unique_dates) or test_mask.sum() < 0.05*len(unique_dates):
            continue

        # Calculate label proportions in each split
        y_train_prop = (y[train_mask] == 1).mean()
        y_test_prop = (y[test_mask] == 1).mean()

        if np.isclose(y_train_prop, 0) or np.isclose(y_test_prop, 0):
            continue

        # combined deviation in label proportions
        combined_score = beta*(abs(y_train_prop - overall_prop)**2 + abs(y_test_prop - overall_prop)**2)
        # deviation in proportion of training samples
        combined_score += (1-beta)*(abs(sum(train_mask)/len(dates) - train_prop)**2 + abs(sum(test_mask)/len(dates) - (1-train_prop))**2)
        if combined_score <= best_score:
            best_score = combined_score
            best_cutoff = cutoff_date

    return best_cutoff