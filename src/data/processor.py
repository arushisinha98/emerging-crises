import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from datasets import load_dataset
import matplotlib.pyplot as plt

import os
import dotenv
dotenv.load_dotenv()
username = os.getenv("HUGGINGFACE_USERNAME")

from .log_utilities import _find_project_root, setup_logging, DataLogger

root_dir = _find_project_root()

class PreprocessPipeline:
    """
    A pipeline for preprocessing financial data with various column/row dropping
    and imputation methods.
    
    This class provides methods for filling missing values in financial
    datasets using various techniques including forward fill, backward fill,
    ARIMA modeling, and KNN-based multiple imputation without look-ahead bias.
    
    Attributes:
    -----------
    data_tag (str)
        Identifier for the pipeline. If this is a data source,
        the appropriate data will be loaded from Hugging Face Hub.
    countries (str)
        Specifies the subset of countries to process (i.e., "developed", "emerging").
    df (pd.DataFrame)
        DataFrame to process. If None, data will be loaded from Hugging Face Hub.
    logger
        Logger instance for logging messages.
    preprocess_log (defaultdict)
        Log of preprocessing operations.
    """
    
    def __init__(self, data_tag: str, subset: str = None, split: str = 'train', df: Optional[pd.DataFrame] = None):
        """Initialize PreprocessPipeline for a DataObject.
        
        Parameters:
        -----------
        data_tag: str
            Name identifier for the pipeline.
        subset: str
            Subset of countries to process (i.e., "developed", "emerging").
        """
        # set up logging to track pipeline
        self.logger = setup_logging(f"{self.__class__.__name__}_logs.txt")

        # attributes for loading data from Hugging Face Hub
        self.data_tag = data_tag
        self.subset = subset

        # if data is provided, add to pipeline
        if df is not None and not isinstance(df, pd.DataFrame):
            raise ValueError("Provided df must be a pandas DataFrame.")
        elif isinstance(df, pd.DataFrame):
            self.df = df.copy()
        
        if df is None:
            try:
                datalink = f"{username}/{data_tag.lower()}-download"
                self.df = self._load_data(datalink, subset, split=split)
                self.logger.info(f"PreprocessPipeline initialized for source: {data_tag}")
            except Exception as e:
                self.logger.error(f"Failed to initialize PreprocessPipeline: {e}")
                raise e

        # preprocess log to store preprocessing steps
        self.preprocess_log = []

    def _load_data(self, datalink, subset = None, split = 'train') -> pd.DataFrame:
        """Load dataset from the specified datalink and set index."""
        try:
            if subset:
                df = load_dataset(datalink, subset, split=split).to_pandas()
            else:
                df = load_dataset(datalink, split=split).to_pandas()
            
            # Convert date column to datetime
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

            # Remove any columns that start with __index_level_
            index_cols_to_remove = [col for col in df.columns if col.startswith('__index_level_')]
            if index_cols_to_remove:
                df.drop(columns=index_cols_to_remove, inplace=True)

            # Drop missing columns, sort index
            if df.shape[0] > 0:
                df.dropna(axis=1, how='all', inplace=True)
            df = df.sort_index()
            if 'Country' in df.columns:
                df = df.sort_values(['Country', 'Date'])
            else:
                df = df.sort_values('Date')

            self.logger.info(f"Loaded dataset from {datalink} with shape {df.shape}")
            return df
        
        except Exception as e:
            self.logger.error(f"Failed to load dataset from {datalink}: {e}")
            return None

    def _completeness(self) -> pd.DataFrame:
        """
        Calculate completeness percentage for each column.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns 'column', 'completeness', and 'missing_percent'.
        """
        completeness = 1 - self.df.isnull().mean()
        completeness_df = completeness.reset_index()
        completeness_df.columns = ['column', 'completeness']
        completeness_df['missing_percent'] = 1 - completeness_df['completeness']
        return completeness_df.sort_values('completeness')
    
    def trim_timeseries(self, completeness: float = 0.4):
        """
        Trim the DataFrame to only include dates with complete columns above a threshold.
        
        Parameters:
        -----------
        completeness: float
            Minimum completeness threshold for columns.

        Returns:
        --------
        pd.DataFrame
            DataFrame with trimmed columns.
        """
        # Order by date
        self.df.sort_values(by='Date', inplace=True)

        # Find first date with completeness above threshold
        complete_index = self.df.isnull().cumsum(axis=1).max(axis=1) < (1-completeness)*self.df.shape[1]
        self.df = self.df[complete_index.cumsum() >= 1]
        self.df.reset_index(drop=True, inplace=True)
        self.logger.info(f"Trimmed DataFrame to dates from {self.df['Date'].min()}")
        
        # Remove 2024 onwards
        keep = self.df['Date'].dt.year < 2024
        self.df = self.df.loc[keep]

        # Log the operation
        self.preprocess_log.append(f"trim_timeseries({completeness})")
        return self

    def drop_columns(self, columns: List[str] = [], completeness=0.6):
        """
        Drop specified columns from the dataframe and/or drop columns by
         completeness threshold.
        
        Parameters:
        -----------
        completeness: float
            Minimum completeness threshold for columns.
        columns: List[str]
            List of column names to drop from the DataFrame.
        """
        # Drop columns from the DataFrame
        self.df.drop(columns=columns, inplace=True, errors='ignore')
        self.logger.info(f"Dropped columns: {columns}")

        # Drop columns with less than completeness threshold
        columns_to_drop = []
        if completeness > 0:
            completeness_df = self._completeness()
            columns_to_drop = completeness_df[completeness_df['completeness'] < completeness]['column'].tolist()
            self.df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
            self.logger.info(f"Dropped {len(columns_to_drop)} columns below completeness threshold {completeness}")
        
        # Log the operation
        self.preprocess_log.append(f"drop_columns(columns={columns}, completeness={completeness})")
        return self

    def drop_static(self):
        """
        Drop static rows/columns from the DataFrame.
        """
        # Identify static columns
        static_cols = self.df.columns[self.df.nunique(dropna=False) <= 2]
        self.df = self.df.drop(columns=static_cols)

        # Identify static rows
        static_rows = self.df.index[self.df[self.df.columns[~self.df.columns.isin(
            ['Date', 'Country'])]].nunique(axis=1) <= 1]
        self.df = self.df.drop(index=static_rows)
        
        self.logger.info(f"{len(static_cols)} static columns removed")
        self.logger.info(f"{len(static_rows)} static rows removed")

        # Log the operation
        self.preprocess_log.append("drop_static()")
        return self

    def resample(self, frequency: str = 'M', mapping: Optional[Dict[str, List[str]]] = None):
        """
        Aggregate the DataFrame to a specified frequency.
        
        Parameters:
        -----------
        frequency: str
            Frequency to aggregate to (e.g., 'M' for monthly).
        mapping: Dict[str, List[str]]
            Optional mapping of columns to aggregate functions (e.g. sum, last, min, max, mean, median).
            If unspecified, will take last value in aggregated period.
        """
        if not mapping:
            mapping = {'last': [col for col in self.df.columns if col not in ['Country', 'Date']]}
        error = [key for key in mapping.keys() if key not in ['sum', 'last', 'min', 'max', 'mean', 'median']]
        if error:
            raise ValueError(f"Mapping keys must be one of: 'sum', 'last', 'min', 'max', 'mean', 'median'. Invalid keys: {error}")

        # Group by country then resample by frequency
        if 'Country' in self.df.columns:
            max_df = self.df.groupby('Country').resample(frequency, on='Date').max(numeric_only=True)
            min_df = self.df.groupby('Country').resample(frequency, on='Date').min(numeric_only=True)
            sum_df = self.df.groupby('Country').resample(frequency, on='Date').sum(numeric_only=True)
            mean_df = self.df.groupby('Country').resample(frequency, on='Date').mean(numeric_only=True)
            median_df = self.df.groupby('Country').resample(frequency, on='Date').median(numeric_only=True)
            last_df = self.df.set_index('Date').sort_index()\
                .groupby('Country').resample(frequency).ffill()\
                    .drop(columns=['Country'])
        # If no Country column, just resample by frequency
        else:
            max_df = self.df.resample(frequency, on='Date').max()
            min_df = self.df.resample(frequency, on='Date').min()
            sum_df = self.df.resample(frequency, on='Date').sum()
            mean_df = self.df.resample(frequency, on='Date').mean()
            median_df = self.df.resample(frequency, on='Date').median()
            last_df = self.df.set_index('Date').sort_index().resample(frequency).ffill()

        # Plug aggregations based on mapping
        for aggregate, columns in mapping.items():
            for col in columns:
                if col in self.df.columns and col not in ['Country', 'Date']:
                    if aggregate == 'sum':
                        last_df[col] = sum_df[col]
                    elif aggregate == 'last':
                        last_df[col] = last_df[col]
                    elif aggregate == 'min':
                        last_df[col] = min_df[col]
                    elif aggregate == 'max':
                        last_df[col] = max_df[col]
                    elif aggregate == 'mean':
                        last_df[col] = mean_df[col]
                    elif aggregate == 'median':
                        last_df[col] = median_df[col]
                    else:
                        raise ValueError(f"Unsupported aggregation type: {aggregate}")
        self.df = last_df.reset_index()
    
        # Get the full date range across all countries
        min_date = self.df['Date'].min()
        max_date = self.df['Date'].max()
        
        # Create complete date range
        if frequency == 'D':
            return self
        elif frequency == 'Y':
            date_range = pd.date_range(start=min_date, end=max_date, freq='YE')  # Year end
        elif frequency == 'M':
            date_range = pd.date_range(start=min_date, end=max_date, freq='ME')  # Month end
        
        # Get all unique countries
        countries = self.df['Country'].unique()
        
        # Create complete country-date combinations
        complete_index = pd.MultiIndex.from_product(
            [countries, date_range], 
            names=['Country', 'Date']
        ).to_frame(index=False)
        
        # Merge with original data to fill gaps
        complete_df = complete_index.merge(
            self.df, 
            on=['Country', 'Date'], 
            how='left'
        )
        
        # Sort by country and date
        self.df = complete_df.sort_values(['Country', 'Date']).reset_index(drop=True)

        self.logger.info(f"Aggregated DataFrame to frequency: {frequency}")

        # Log the operation
        self.preprocess_log.append(f"aggregate(frequency={frequency}, mapping={mapping})")
        return self

    def forward_fill(self, n: int = None, latest_only: bool = True,
                     columns: Optional[List[str]] = None):
        """
        Fill up to n missing values in specified columns using forward fill method.
        
        If no columns are specified, forward fill all columns.
        If latest_only is True, only fill up to n latest values, otherwise 
        forward fill up to n values between reported values.
        Can be used for data reporting lag or increasing resolution of aggregates.
        If 'Country' column is present, fill for each country separately.

        Parameters:
        -----------
        n: int
            Maximum number of values to fill. If None, fills all NaNs.
        latest_only: bool
            If True, only fill latest missing values.
        columns: List[str]
            List of columns to process. If None, processes all columns.
        """
        columns = [col for col in (columns or self.df.columns) if col in self.df.columns and col not in ['Country', 'Date']]
        if 'Country' not in self.df.columns:
            self.df['Country'] = 'N/A'
        
        self.df = self.df.sort_values(['Country', 'Date'])
        countries = self.df['Country'].unique()
        n = n if n is not None else self.df.shape[0]  # Default to fill all rows

        initial_missing = self.df.isnull().sum().sum()

        if latest_only:
            for country in countries:
                country_df = self.df[(self.df['Country'] == country)].sort_values(by='Date')
                for col in columns:
                    last_valid = country_df[[col]].last_valid_index()
                    if last_valid is not None:
                        # Create mask for last valid index and after
                        mask = country_df.index[(country_df.index == last_valid).cumsum() == 1]
                        # Fill the next n missing values with last observation
                        filled_slice = country_df.loc[mask, col].ffill(limit=n)
                        self.df.loc[filled_slice.index, col] = filled_slice.values
        else:
            self.df[columns] = self.df.sort_values(by='Date').groupby('Country')[columns].ffill(limit=n)

        if 'N/A' in countries:
            self.df.drop(columns=['Country'], inplace=True)
        
        filled_values = initial_missing - self.df.isnull().sum().sum()
        self.logger.info(f"Forward-fill completed: {filled_values} values filled.")

        # Log the operation
        self.preprocess_log.append(f"forward_fill(n={n}, latest_only={latest_only}, columns={columns})")
        return self

    def zero_fill(self):
        """
        Fill a column with 0 if it was never reported or if all values are 0.
        """
        self.df = self.df.sort_values(['Country', 'Date'])

        columns = [col for col in self.df.columns if col not in ['Country', 'Date']]
        if 'Country' not in self.df.columns:
            self.df['Country'] = 'N/A'
        countries = self.df['Country'].unique()

        initial_missing = self.df.isnull().sum().sum()
        countries_filled, columns_filled = set(), set()

        for country in countries:
            country_df = self.df[(self.df['Country'] == country)].sort_values(by='Date')
            for col in columns:
                non_null = country_df[col].dropna()
                # Fill with 0 if all values are NaN OR if all non-null values are 0
                if country_df[col].isnull().all() or (len(non_null) > 0 and (non_null == 0).all()):
                    self.df.loc[country_df.index, col] = 0
                    countries_filled.add(country)
                    columns_filled.add(col)

        if 'N/A' in countries:
            self.df.drop(columns=['Country'], inplace=True)
        
        filled_values = initial_missing - self.df.isnull().sum().sum()
        self.logger.info(f"Zero-fill completed: {filled_values} values filled for {countries_filled}, {columns_filled}")
        
        # Log the operation
        self.preprocess_log.append("zero_fill()")
        return self

    def backfill(self, fill: str = "mean",
                 columns: Optional[List[str]] = None):
        """
        Backfill missing values with zero or statistical measures.
        
        Parameters:
        -----------
        df: pd.DataFrame
            DataFrame to process.
        fill: str
            Fill method ('zero', 'mean', 'median', etc.).
        columns: List[str]
            List of columns to process. If None, processes all columns.
        """
        if fill not in ["mean", "median", "mode", "zero"]:
            raise ValueError(f"Unsupported fill method: {fill}. "
                            "Supported methods are 'mean', 'median', 'mode', 'zero'.")
        
        columns = [col for col in (columns or self.df.columns) if col in self.df.columns and col not in ['Country', 'Date']]
        if 'Country' not in self.df.columns and fill != "zero":
            self.logger.warning("No 'Country' column found. Cannot compute global statistics to backfill.")
            return self
        
        elif 'Country' not in self.df.columns:
            self.df['Country'] = 'N/A'
        
        self.df = self.df.sort_values(['Country', 'Date'])
        countries = self.df['Country'].unique()
        initial_missing = self.df.isnull().sum().sum()
        
        for country in countries:
            country_df = self.df[(self.df['Country'] == country)].sort_values(by='Date')
            for col in columns:
                # Find the first index with non-null value
                first_valid = country_df[col].first_valid_index()
                if first_valid is not None:
                    # Create mask for up to first valid index
                    mask = country_df.index[(country_df.index == first_valid).cumsum() == 0]
                    if fill == "zero":
                        self.df.loc[mask, col] = 0
                    else:
                        # Fill with global statistic
                        global_val = self.df.groupby('Date')[col].transform(fill)
                        self.df.loc[mask, col] = global_val.loc[mask].values

        if 'N/A' in countries:
            self.df.drop(columns=['Country'], inplace=True)
        
        filled_values = initial_missing - self.df.isnull().sum().sum()
        self.logger.info(f"Backfill completed: {filled_values} values filled.")
        
        # Log the operation
        self.preprocess_log.append(f"backfill(fill={fill}, columns={columns})")
        return self

    def get_imputation_mask(self, columns: Optional[List[str]] = None,
                            include_before: bool = False,
                            include_after: bool = False) -> pd.DataFrame:
        """
        Get a mask DataFrame indicating missing values in specified columns.
        
        Parameters:
        -----------
        columns: List[str]
            List of columns to check for missing values. If None, checks all columns.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with boolean values indicating missing data.
        """
        columns = [col for col in (columns or self.df.columns) if col in self.df.columns and col not in ['Country', 'Date']]
        if 'Country' not in self.df.columns:
            self.df['Country'] = 'N/A'
        self.df = self.df.sort_values(['Country', 'Date'])
        countries = self.df['Country'].unique()

        mask = pd.DataFrame(index=self.df.index, columns=columns, dtype=bool)
        mask.fillna(False, inplace=True)

        for ii, col in enumerate(columns):
            for country in countries:
                country_mask = self.df['Country'] == country
                series = self.df.loc[country_mask, col]
                is_null = series.isnull()
                
                # Case 0: entire series is filled, no interpolation
                if not is_null.any():
                    mask.loc[series.index, col] = False
                    continue

                # Case 1: entire series is null, no interpolation
                if is_null.all():
                    mask.loc[series.index, col] = False
                    continue

                # Case 2: series has some valid values
                start = series.first_valid_index()
                end = series.last_valid_index()

                if start is not None and end is not None and start < end:
                    start_end_slice = series.loc[start:end].index
                    is_null_slice = is_null.loc[start:end].fillna(False).astype(bool)
                    mask.loc[start_end_slice, col] = is_null_slice.values

                    if start != series.index[0]: # if first valid value is not the beginning of the series
                        before_slice = series.loc[:start].index[:-1]  # exclude first valid index
                    else:
                        before_slice = []
                    
                    if end != series.index[-1]: # if last valid value is not the end of the series
                        after_slice = series.loc[end:].index[1:]
                    else:
                        after_slice = []
                    
                    if include_before: # include null values before the first valid value
                        if len(before_slice) > 0:
                            is_null_before = is_null.loc[before_slice].fillna(False).astype(bool)
                            mask.loc[before_slice, col] = is_null_before.values
                    else: # do not include any null values before the first valid value
                        if len(before_slice) > 0:
                            mask.loc[before_slice, col] = False

                    if include_after: # include null values after the last valid value
                        if len(after_slice) > 0:
                            is_null_after = is_null.loc[after_slice].fillna(False).astype(bool)
                            mask.loc[after_slice, col] = is_null_after.values
                    else: # do not include any null values after the last valid value
                        if len(after_slice) > 0:
                            mask.loc[after_slice, col] = False
                
                # Case 3: series has one valid value, no interpolation
                else:
                    mask.loc[series.index, col] = False

        if 'N/A' in countries:
            self.df.drop(columns=['Country'], inplace=True)
        
        mask_true_count = mask.sum().sum()
        self.logger.info(f"Generated imputation mask with {mask_true_count} values to impute across {len(columns)} columns.")

        # Log the operation
        self.preprocess_log.append(f"get_imputation_mask(columns={columns}, include_before={include_before}, include_after={include_after})")
        return mask
    
    def knn_iterative_imputer(self, columns: Optional[List[str]] = None, 
                              k: int = 5,
                              mask: Optional[pd.DataFrame] = None,
                              feature_columns: Optional[List[str]] = None,
                              distance_metric: str = 'euclidean',
                              normalize_features: bool = True):
        """
        Perform KNN-based imputation without look-ahead bias.
        
        For each missing value, only uses data from the same date or earlier dates
        to find k nearest neighbors and impute based on their average.
        
        Parameters:
        -----------
        columns: List[str]
            List of columns to impute. If None, imputes all numeric columns.
        k: int
            Number of nearest neighbors to use for imputation.
        mask: Optional[pd.DataFrame]
            Imputation mask indicating which values to impute. If None,
            imputes all missing values.
        feature_columns: Optional[List[str]]
            Columns to use for computing distances. If None,
            uses all numeric columns except those being imputed.
        distance_metric: str
            Distance metric for KNN ('euclidean', 'manhattan', etc.)
        normalize_features: bool
            Whether to normalize features before computing distances.
        """
        self.df = self.df.sort_values(['Country', 'Date']).reset_index(drop=True)
        
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            columns = [col for col in numeric_cols if col not in ['Country', 'Date']]
        else:
            columns = [col for col in columns if col in self.df.columns and col not in ['Country', 'Date']]

        # Determine feature columns for distance computation
        if feature_columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numeric_cols if col not in ['Country', 'Date']]
        
        if mask is None:
            mask = self.df[columns].isna()
        else:
            mask = mask.reindex(index=self.df.index, columns=columns, fill_value=False)

        imputation_count = 0

        # Process each row that needs imputation
        for idx in self.df.index:
            current_row = self.df.loc[idx]
            current_date = current_row['Date']
            
            # Check if this row needs any imputation
            needs_imputation = mask.loc[idx].any() if isinstance(mask.loc[idx], pd.Series) else mask.loc[idx]
            if not needs_imputation:
                continue
                
            # Get historical data (up to and including current date)
            historical_data = self.df[self.df['Date'] <= current_date].copy()
            
            # Remove the current row from historical data to avoid self-matching
            historical_data = historical_data.drop(idx, errors='ignore')
            
            if len(historical_data) == 0:
                # self.logger.warning(f"No historical data available for imputation at index {idx}")
                continue
            
            # Prepare feature matrix for distance computation
            feature_data = historical_data[feature_columns].copy()
            current_features = current_row[feature_columns].copy()
            
            # Get features available in current row
            available_features = current_features.dropna().index.tolist()

            if len(available_features) == 0:
                # self.logger.warning(f"No available features for distance computation at index {idx}")
                continue

            # Subset to only use available features
            feature_data = feature_data[available_features].copy()
            current_features = current_features[available_features].copy()

            # Drop rows missing these specific features
            complete_mask = feature_data.notna().all(axis=1)
            feature_data = feature_data[complete_mask]
            historical_complete = historical_data[complete_mask]
            
            if len(feature_data) == 0:
                # self.logger.warning(f"No historical features available for imputation at index {idx}")
                continue
            
            # Normalize features
            if normalize_features:
                scaler = RobustScaler()
                try:
                    # Fit and transform historical data
                    feature_data_scaled = scaler.fit_transform(feature_data)
                    
                    # Transform current features
                    current_features_df = pd.DataFrame([current_features], columns=available_features)
                    current_features_scaled = scaler.transform(current_features_df)
                    
                except ValueError as e:
                    # self.logger.warning(f"Scaling failed at index {idx}: {e}")
                    continue
            else:
                feature_data_scaled = feature_data.values
                current_features_scaled = current_features.values.reshape(1, -1)
            
            # Find k nearest neighbors
            effective_k = min(k, len(feature_data_scaled))
            
            try:
                knn = NearestNeighbors(n_neighbors=effective_k, metric=distance_metric)
                knn.fit(feature_data_scaled)
                distances, neighbor_indices = knn.kneighbors(current_features_scaled)
            except Exception as e:
                # self.logger.warning(f"KNN failed at index {idx}: {e}")
                continue
            
            # Get the actual neighbor rows
            neighbor_rows = historical_complete.iloc[neighbor_indices[0]]
            
            # Impute each column that needs imputation
            for col in columns:
                if mask.loc[idx, col]:  # Only impute if mask indicates it should be imputed
                    neighbor_values = neighbor_rows[col].dropna()
                    
                    if len(neighbor_values) > 0:
                        # Get the indices of non-null neighbors
                        valid_neighbor_mask = neighbor_rows[col].notna()
                        # Use weighted average based on inverse distance
                        valid_weights = distances[0][valid_neighbor_mask]
                        weights = 1 / (valid_weights + 1e-8) # Add small epsilon to avoid division by zero
                        
                        if len(weights) > 0:
                            imputed_value = np.average(neighbor_values, weights=weights)
                            self.df.loc[idx, col] = imputed_value
                            imputation_count += 1
                        # else:
                        #     self.logger.warning(f"No valid neighbor values for column {col} at index {idx}")
                    # else:
                    #     self.logger.warning(f"No non-null neighbor values for column {col} at index {idx}")

        self.logger.info(f"KNN imputation completed. Imputed {imputation_count} values using k={k} neighbors.")

        # Log the operation
        self.preprocess_log.append(f"knn_iterative_imputer(columns={columns}, k={k}, "
                                   f"feature_columns={feature_columns}, distance_metric='{distance_metric}', "
                                   f"normalize_features={normalize_features})")
        return self

    @staticmethod
    def select_arima_order(observed, max_p=5, max_d=2, max_q=5):
        """
        Select ARIMA order using AIC with constraints.
        
        Parameters:
        -----------
        observed: pd.Series
            Time series data to fit ARIMA model to.
        max_p: int
            Maximum autoregressive order to consider.
        max_d: int
            Maximum differencing order to consider.
        max_q: int
            Maximum moving average order to consider.
            
        Returns:
        --------
        Tuple
            Best (p, d, q) order based on AIC criterion.
        """
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        best_aic = np.inf
        best_order = (0, 0, 0)
        
        # Determine d first
        d = 0
        temp = observed.copy()
        result = adfuller(temp)
        while result[1] > 0.05 and d < max_d:
            temp = temp.diff().dropna()
            d += 1
            if len(temp) > 3:
                result = adfuller(temp)
            else:
                break
        
        # Search over p,q combinations
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(observed, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except:
                    continue
        
        return best_order

    def arima_fill(self, columns: Optional[List[str]] = None) -> 'PreprocessPipeline':
        """
        Fill missing values in specified columns using ARIMA model.
        
        Parameters:
        -----------
        columns: List[str]
            List of columns to process. If None, processes all numeric columns.
        """
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        if columns is None:
            columns = [col for col in self.df.columns if col not in ['Country', 'Date']]
        else:
            columns = [col for col in columns if col in self.df.columns and col not in ['Country', 'Date']]
        
        countries = self.df['Country'].unique()
        initial_missing = self.df.isnull().sum().sum()
        fitted_models = 0
        processed_series = 0
        
        for col in columns:
            for country in countries:
                country_mask = self.df['Country'] == country
                country_data = self.df.loc[country_mask, col].copy().reset_index(drop=True)
                original_indices = self.df.loc[country_mask, col].index

                if country_data.isna().all():
                    continue
                
                # Process all missing segments for country-column combination
                while True:
                    # Find next missing segment
                    first_valid = country_data.first_valid_index()
                    if first_valid is None:
                        break
                    
                    # Find missing values after first valid
                    missing_after_valid = country_data.loc[first_valid:].isna()
                    if not missing_after_valid.any():
                        break
                    
                    first_invalid = missing_after_valid[missing_after_valid].index[0]
                    
                    # Get observed data for ARIMA
                    observed = country_data.loc[first_valid:first_invalid].dropna()
                    if len(observed) < 6:
                        break
                    
                    # Handle constant series
                    if len(observed.unique()) == 1:
                        self.df.loc[country_mask & self.df[col].isna(), col] = observed.iloc[0]
                        break
                    
                    # Find next valid value to determine fill range
                    next_valid = country_data.loc[first_invalid:].first_valid_index()
                    if next_valid is None:
                        # Fill to end of series
                        fill_idx = country_data.loc[first_invalid:].index
                    else:
                        # Fill between invalid and next valid
                        fill_idx = country_data.loc[first_invalid:next_valid].index[:-1]
                    
                    if len(fill_idx) == 0:
                        break
                    
                    try:
                        # Fit ARIMA and predict
                        p, d, q = self.select_arima_order(observed)
                        model = ARIMA(observed, order=(p, d, q))
                        fitted = model.fit()
                        fitted_models += 1
                        
                        # Get predictions
                        pred = fitted.forecast(steps=len(fill_idx))
                        
                        # Fill missing values
                        for i, idx in enumerate(fill_idx):
                            original_idx = original_indices[country_data.index.get_loc(idx)]
                            self.df.loc[original_idx, col] = pred.iloc[i]
                        
                        # Update country_data for next iteration
                        country_data = self.df.loc[country_mask, col].copy()
                        
                    except Exception as e:
                        self.logger.warning(f"ARIMA fitting failed for {country}-{col}: {e}")
                        break
                
                processed_series += 1
        
        final_missing = self.df.isnull().sum().sum()
        filled_values = initial_missing - final_missing
        
        self.logger.info(f"ARIMA-fill completed: {filled_values} values filled using {fitted_models} fitted models across {processed_series} series")
        
        # Log the operation
        self.preprocess_log.append(f"arima_fill(columns={columns})")
        return self
    
    def print_info(self, countries: List[str] = None) -> None:
        """
        Print data quality information in a format suitable for markdown table.
        Prints: Data Set, Number of Countries, Average Temporal Range, Average Completeness
        """
        if countries is not None and len(countries) > 0:
            sub_df = self.df.loc[self.df['Country'].isin(countries)]
        else:
            sub_df = self.df

        if sub_df.empty:
            return ""
        
        exclude_cols = ['Country', 'Date']
        data_cols = [col for col in sub_df.columns if col not in exclude_cols]
        
        # Basic statistics
        total_countries = len(sub_df['Country'].unique()) if 'Country' in sub_df.columns else 1
        
        # Calculate temporal range based on when at least half the countries have valid data
        if 'Country' in sub_df.columns and total_countries > 0:
            all_dates = pd.to_datetime(sub_df['Date']).dropna()
            if len(all_dates) > 0:
                avg_temporal_range = f"{all_dates.min().year}-{all_dates.max().year}"
            else:
                avg_temporal_range = "N/A"
        
        if data_cols:
            # Calculate completeness across all data columns for the subset
            total_completeness = (1 - sub_df[data_cols].isnull().mean().mean()) * 100
        else:
            total_completeness = 0.0
        
        # Print in table format
        print(f"| {self.data_tag} | {total_countries} | {avg_temporal_range} | {total_completeness:.1f}% |")

    def plot_sample(self, countries: List[str] = ['United States', 'Japan']):
        """
        Plot a sample of time series data, highlighting selected countries.
        """
        # Get data columns (exclude Country and Date)
        exclude_cols = ['Country', 'Date']
        data_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        np.random.seed(42)
        if 'Country' not in self.df.columns:
            self.df['Country'] = 'N/A'

        # Randomly sample 9 columns as backup
        if len(data_cols) == 0:
            print("No data columns available to plot.")
            return

        # ROW 1: Plot 3 examples of columns with gaps in the middle
        # ROW 2: Plot 3 examples of columns with gaps before
        # ROW 3: Plot 3 examples of columns with gaps after
        row1_columns, row2_columns, row3_columns = [], [], []

        for col in data_cols:
            for country in self.df['Country'].unique():
                country_series = self.df[self.df['Country'] == country].sort_values('Date')[col]
                if country_series.isna().all():
                    continue
                
                first_valid = country_series.first_valid_index()
                last_valid = country_series.last_valid_index()

                if first_valid > country_series.index[0]:
                    row2_columns.append(col)
                if last_valid < country_series.index[-1]:
                    row3_columns.append(col)

                if first_valid is not None and last_valid is not None:
                    between_range = country_series.loc[first_valid:last_valid]
                    if between_range.isna().any():
                        row1_columns.append(col)
        
        # Randomly sample 3 from each (if available)
        sample_size = min(3, len(row1_columns))
        plot_cols = np.random.choice(row1_columns, size=sample_size, replace=False).tolist()

        sample_size = min(3, len(row2_columns))
        if sample_size > 0:
            plot_cols.extend(np.random.choice(row2_columns, size=sample_size, replace=False).tolist())

        sample_size = min(3, len(row3_columns))
        if sample_size > 0:
            plot_cols.extend(np.random.choice(row3_columns, size=sample_size, replace=False).tolist())

        # If not enough columns, use backup sample
        backup_samples = np.random.choice(data_cols, size=len(data_cols), replace=False)
        backup_samples = [col for col in backup_samples if col not in plot_cols]
        if len(plot_cols) < 9 and len(backup_samples) >= 9 - len(plot_cols): # Fallback to backup sample
            for _ in range(9 - len(plot_cols)):
                plot_cols.append(backup_samples[0])
                backup_samples = backup_samples[1:]
        
        # Create figure with 3x3 subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot each sampled column
        for i, col in enumerate(plot_cols):
            ax = axes[i]
            
            for country in self.df['Country'].unique():
                # Get country-specific data
                country_subset = self.df[self.df['Country'] == country].sort_values('Date')
                dates = pd.to_datetime(country_subset['Date'])
                values = country_subset[col]
                
                if country in countries:
                    ax.plot(dates, values, label=country, linewidth=1.5, alpha=0.8)
                else:
                    ax.plot(dates, values, linewidth=0.5, alpha=0.5, color='k')
            
            ax.set_title(f'{col}', fontsize=10)
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)
            all_dates = pd.to_datetime(self.df['Date'])
            ax.set_xlim(all_dates.min(), all_dates.max())
            if i == 0:
                ax.legend()
        
        # Hide unused subplots if less than 9 columns
        for i in range(len(plot_cols), 9):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        if 'N/A' in countries:
            self.df.drop(columns=['Country'], inplace=True)

    def multiple_imputation(self, df: pd.DataFrame, regressor,
                            mask: pd.DataFrame,
                            features: pd.DataFrame = None):
        """
        Perform multiple imputation using a specified scikit-learn regressor.
        
        e.g. RandomForestRegressor, HistGradientBoostingRegressor
        Prevents look-ahead bias by only using past data for each imputation.
        
        Parameters:
        -----------
        df: pd.DataFrame
            DataFrame to process.
        regressor: Scikit-learn regressor
            Regressor instance for imputation.
        mask: pd.DataFrame
            Boolean mask indicating which values to impute.
        features: pd.DataFrame
            Optional additional features DataFrame.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with imputed values.
        """
        from sklearn.base import clone

        with DataLogger(self.logger, f"multiple_imputation(regressor={type(regressor).__name__})", df_shape=df.shape):
            df_sorted = df.sort_values('Date')
            time_points = sorted(df_sorted['Date'].unique())
            df_imputed = df_sorted.copy()
            
            initial_missing = df.isnull().sum().sum()
            trained_models = 0
            total_time_points = len(time_points)
            
            for current_time in time_points:
                # Get historical mask
                historical_mask = df_sorted[df_sorted['Date'] < current_time].index
                historical_data = df_sorted.loc[historical_mask].copy()
                current_time_data = df_sorted.xs(current_time, level=1)
                current_time_mask = mask.xs(current_time, level=1)
                missing_at_current_time = current_time_data.isnull()
                if not missing_at_current_time.any().any():
                    continue

                # Prepare features
                df_features = historical_data.copy()
                # Label encode Country
                df_features['Country-Encoded'] = df_features['Country'].astype('category').cat.codes
                # Convert temporal index to time since baseline
                dates = pd.to_datetime(historical_data.index.get_level_values(1))
                baseline_date = dates.min()
                df_features['days_since_baseline'] = (dates - baseline_date).days
                df_features['year'] = dates.year
                df_features['month'] = dates.month
                
                # Add external features if provided
                if features is not None:
                    historical_features = features.loc[historical_mask]
                    df_features = pd.concat([df_features, historical_features], axis=1)
                    df_features = df_features.loc[:,~df_features.columns.duplicated()].copy()
                
                # For each column with missing values at current time
                # train imputer on historical data
                for col in df.columns:
                    countries_missing = missing_at_current_time[
                        missing_at_current_time[col] & current_time_mask[col]].index
                    
                    if len(countries_missing) == 0:
                        continue
                        
                    feature_cols = [c for c in df_features.columns if c != col]
                    training_mask = ~historical_data[col].isnull()
                    
                    if training_mask.sum() < 10: # Minimum training samples
                        continue
                        
                    X_train = df_features.loc[training_mask, feature_cols]
                    y_train = historical_data.loc[training_mask, col]
                    prediction_indices = [(country, current_time) \
                                          for country in countries_missing]
                    X_predict = df_features.loc[prediction_indices, feature_cols]
                    
                    # Train and predict
                    try:
                        temp_regressor = clone(regressor)
                        temp_regressor.fit(X_train, y_train)
                        trained_models += 1
                        predictions = temp_regressor.predict(X_predict)
                        for i, country in enumerate(countries_missing):
                            df_imputed.loc[(country, current_time), col] = predictions[i]
                                
                    except Exception as e:
                        continue
            
            final_missing = df_imputed.isnull().sum().sum()
            filled_values = initial_missing - final_missing
            
            self.logger.info(f"Multiple imputation completed: {filled_values} values filled using {trained_models} trained models across {total_time_points} time points.")
            
        return df_imputed
    
    def engineer_features(self, df: pd.DataFrame,
                          window: int = 3) -> pd.DataFrame:
        """
        Engineer additional features from the existing DataFrame.
        
        Creates features including rolling volatility, momentum indicators,
        exponentially weighted moving averages, and global averages.
        
        Parameters:
        -----------
        df: pd.DataFrame
            DataFrame to process.
        window: int
            Window size for rolling calculations.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with engineered features (original columns dropped).
        """
        with DataLogger(self.logger, f"engineer_features(window={window})", df_shape=df.shape):
            df_features = df.copy()

            data_cols = [col for col in df.columns if col not in ['Country', 'Date']]
            for col in data_cols:
                # Feature 1: Rolling Volatility
                # captures the stability/instability of each indicator within countries
                rolling_vol = df.groupby('Country')[col].sort_values('Date').rolling(window=window).std()
                df_features[f'{col}_{window}_year_volatility'] = rolling_vol.values
                # Feature 2: Change detection with rolling mean
                # detects whether indicators are trending upward or downward
                short_ma = df.groupby('Country')[col].sort_values('Date').rolling(window=max(1, window-1)).mean()
                long_ma = df.groupby('Country')[col].sort_values('Date').rolling(window=window+1).mean()
                df_features[f'{col}_momentum'] = short_ma.values - long_ma.values
                # Feature 3: Exponentially weighted moving average (EWMA)
                # captures different time horizons of influence
                for decay in [0.1, 0.3, 0.7]:
                    ewma = df.groupby('Country')[col].sort_values('Date').ewm(alpha=decay).mean()
                    df_features[f'{col}_ewma_{int(decay*10)}'] = ewma.values
                # Feature 4: Global annual average
                # provides global context
                global_avg = df.groupby('Date')[col].mean()
                df_features[f'{col}_global_avg'] = df_features['Date'].map(global_avg)

            # Drop original columns
            df_features = df_features.drop(columns=df.columns, errors='ignore')
            self.logger.info(f"Feature engineering completed: generated {len(df_features.columns)} features.")
            
        return df_features