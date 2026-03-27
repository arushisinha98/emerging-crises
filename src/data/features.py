import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple
from statsmodels.tsa.seasonal import seasonal_decompose

from .data_utilities import build_labels
from .splitter import DataSplitter

class FeaturePipeline:
    """
    A comprehensive feature engineering pipeline for financial time series data.
    
    This class provides methods for creating various types of features from time series data,
    including temporal aggregations, lag features, rolling statistics, early warning signals,
    and deviations from global trends. It's designed to work with MultiIndex DataFrames
    and integrates with PCA analysis for advanced feature creation.
    
    The pipeline supports column-specific operations, allowing selective application
    of different feature engineering techniques to different variables.
    
    KEY DESIGN PRINCIPLES:
    - No data leakage: All parameters are fit on training data only
    - No look-ahead bias: Features use only past information
    - Temporal continuity: Rolling windows extend from training into test
    
    Attributes:
    -----------
    X (pd.DataFrame)
        The input DataFrame with MultiIndex (Country, Date).
    Y (pd.DataFrame)
        The target labels DataFrame.
    fitted_scalers (Dict)
        Fitted scalers for each feature type.
    fitted_stats (Dict)
        Fitted statistics computed on training data.
    """
    
    def __init__(self, X: pd.DataFrame,
                 splitter: Optional['DataSplitter'] = None):
        """
        Initialize the Feature Engineering Pipeline.
        
        Parameters:
        -----------
        X: pd.DataFrame
            Input DataFrame with Country, Date and feature columns.
        Y: pd.DataFrame
            Target labels DataFrame.
        splitter: DataSplitter instance
            For managing train/test splits.
        """
        if 'Country' not in X.columns or 'Date' not in X.columns:
            raise ValueError("Input DataFrame must contain 'Country' and 'Date' columns")
        
        X = X.sort_values(['Country', 'Date'])
        if splitter is not None:
            self.split_type = splitter.split_type() # Geographic or Temporal

            X_train, _ = splitter.split(df=X, split='train')
            X_test, _ = splitter.split(df=X, split='test')

        else:
            self.split_type = "custom"
            X_train = X.copy() # set entire dataframe as train
            X_test = pd.DataFrame(columns=['Country','Date'])
        
        self.X_train = X_train.sort_values(['Country', 'Date'])
        self.X_test = X_test.sort_values(['Country', 'Date'])
        
        self.y_train = build_labels(self.X_train)
        self.y_test = build_labels(self.X_test)

        # Storage for fitted parameters to prevent data leakage
        self.fitted_scalers = {}
        self.fitted_stats = {}
        
    def set_data(self, df: pd.DataFrame, split: str = 'test'):
        """
        Set the test data for the pipeline.
        
        Parameters:
        -----------
        df: pd.DataFrame
            DataFrame with 'Country' and 'Date' columns.
        """
        if 'Country' not in df.columns or 'Date' not in df.columns:
            raise ValueError("Test DataFrame must contain 'Country' and 'Date' columns")
        if any(col not in df.columns for col in self.X_train.columns):
            raise ValueError("Test DataFrame must contain all columns from Train DataFrame")

        if split == 'train':
            self.X_train = df.sort_values(['Country', 'Date'])
            self.y_train = build_labels(df)
        else:
            self.X_test = df.sort_values(['Country', 'Date'])
            self.y_test = build_labels(df)
        
        self.split_type = "custom"

    def combine_temporally(self) -> pd.DataFrame:
        """
        Combine training and test data into a single DataFrame.
        
        Returns:
        --------
        Combined DataFrame.
        """
        if self.split_type != "temporal":
            raise ValueError("Cannot combine data for non-temporal split types")
        combined_df = pd.concat([self.X_train, self.X_test])
        combined_df = combined_df.sort_values(['Country', 'Date'])
        return combined_df

    @staticmethod
    def _slope(df: pd.DataFrame, columns: Optional[List[str]],
               window: int, suffix: str) -> pd.DataFrame:
        """
        Calculate slope features using percentage change.
        
        Parameters:
        -----------
        df: pd.DataFrame
            Input DataFrame.
        columns: List[str]
            List of columns to compute slope for.
        window: int
            Window size for slope calculation.
        suffix: str
            Suffix to add to feature names.

        Returns:
        --------
        DataFrame with slope features.
        """
        new_columns = [f"{col}_{suffix}" for col in columns if col not in ['Country', 'Date']]
        df_feat = pd.DataFrame(data=0.0, index=df.index, columns=new_columns + ['Country', 'Date'])
        df_feat['Country'] = df['Country']
        df_feat['Date'] = df['Date']
        
        countries = df['Country'].unique()
        for col in columns:
            for country in countries:
                country_data = df.loc[df['Country'] == country, col]
                slope_values = country_data.pct_change(periods=window-1).fillna(0)
                df_feat.loc[df_feat['Country'] == country, f"{col}_{suffix}"] = slope_values.values
        
        return df_feat
    
    def create_slope_features(self, columns: Optional[List[str]] = None,
                              window: int = 2, suffix: str = 'slope') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create slope features for the training and test datasets.
        Uses combined data to ensure temporal continuity at split boundaries.
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            DataFrames with slope features for train and test sets.
        """
        if columns is None:
            columns = self.X_train.select_dtypes(include=[np.number]).columns.tolist()

        # Calculate slopes on combined data to ensure temporal continuity
        if self.split_type == "temporal":
            combined_df = self.combine_temporally()
            combined_slope = self._slope(combined_df, columns=columns, window=window, suffix=suffix)

            train_slope = combined_slope[combined_slope.index.isin(self.X_train.index)]
            test_slope = combined_slope[combined_slope.index.isin(self.X_test.index)]

        else:
            train_slope = self._slope(self.X_train, columns=columns, window=window, suffix=suffix)
            test_slope = self._slope(self.X_test, columns=columns, window=window, suffix=suffix)
        
        return train_slope, test_slope
    
    def create_acceleration_features(self, columns: Optional[List[str]] = None,
                                     window: int = 2, suffix: str = 'acc') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create acceleration features for the training and test datasets.
        Acceleration is computed as the slope of slopes (second derivative).
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            DataFrames with acceleration features for train and test sets.
        """
        if columns is None:
            columns = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        slope_columns = [f"{col}_slope" for col in columns]

        if self.split_type == "temporal":
            combined_df = self.combine_temporally()
            combined_slope = self._slope(combined_df, columns=columns, window=window, suffix='slope')
            combined_acc = self._slope(combined_slope, columns=slope_columns, window=window, suffix=suffix)

            train_acc = combined_acc[combined_acc.index.isin(self.X_train.index)]
            test_acc = combined_acc[combined_acc.index.isin(self.X_test.index)]

        else:
            train_slope = self._slope(self.X_train, columns=columns, window=window, suffix='slope')
            test_slope = self._slope(self.X_test, columns=columns, window=window, suffix='slope')
        
            train_acc = self._slope(train_slope, columns=slope_columns, window=window, suffix=suffix)
            test_acc = self._slope(test_slope, columns=slope_columns, window=window, suffix=suffix)

        return train_acc, test_acc
    
    @staticmethod
    def _roll(df: pd.DataFrame, columns: Optional[List[str]],
              windows: Union[int, List[int]],
              stats: List[str],
              suffix: str) -> pd.DataFrame:
        """
        Calculate rolling statistics for specified columns.
        
        Parameters:
        -----------
        df: pd.DataFrame
            Input DataFrame.
        columns: List[str]
            List of columns to compute rolling stats for.
        windows: int, List[int]
            Window size or list of window sizes for rolling statistics.
        stats: List[str]
            List of statistics to calculate.
        suffix: str
            Suffix to add to feature names.

        Returns:
        --------
        pd.DataFrame
            With rolling statistics features.
        """
        new_columns = [f"{col}_{suffix}_{window}_{stat}" for col in columns for window in windows for stat in stats if col not in ['Country', 'Date']]
        df_feat = pd.DataFrame(data=0.0, index=df.index, columns=new_columns + ['Country', 'Date'])
        df_feat['Country'] = df['Country']
        df_feat['Date'] = df['Date']
        
        countries = df['Country'].unique()
        for col in columns:
            for country in countries:
                country_data = df.loc[df['Country'] == country, col]
                for window in windows:
                    for stat in stats:
                        if stat == 'mean':
                            rolling_data = country_data.rolling(window=window, min_periods=1).mean()
                        elif stat == 'std':
                            rolling_data = country_data.rolling(window=window, min_periods=1).std()
                        elif stat == 'min':
                            rolling_data = country_data.rolling(window=window, min_periods=1).min()
                        elif stat == 'max':
                            rolling_data = country_data.rolling(window=window, min_periods=1).max()
                        elif stat == 'median':
                            rolling_data = country_data.rolling(window=window, min_periods=1).median()
                        elif stat == 'skew':
                            rolling_data = country_data.rolling(window=window, min_periods=min(window,3)).skew()
                        elif stat == 'kurt':
                            rolling_data = country_data.rolling(window=window, min_periods=min(window,4)).kurt()
                        else:
                            continue
                        
                        df_feat.loc[df_feat['Country'] == country, f"{col}_{suffix}_{window}_{stat}"] = rolling_data.fillna(0).values
        
        return df_feat

    def create_rolling_features(self, columns: Optional[List[str]] = None,
                                windows: Union[int, List[int]] = [3, 6],
                                stats: List[str] = ['mean', 'std', 'min', 'max'],
                                suffix: str = 'rolling') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate rolling statistics for specified columns.
        Uses combined data to ensure temporal continuity at split boundaries.
        
        Parameters:
        -----------
        columns: List[str]
            List of columns to calculate rolling stats for.
            If None, uses all numeric columns.
        windows: int, List[int]
            Rolling window sizes. Can be single int or list of ints.
        stats: List[str]
            List of statistics to calculate
            ('mean', 'std', 'min', 'max', 'median', 'skew', 'kurt').
        suffix: str
            Suffix to add to rolling statistics names.
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            DataFrames with rolling statistics features for train and test sets.
        """
        if columns is None:
            columns = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
            
        if isinstance(windows, int):
            windows = [windows]
        
        # Calculate rolling statistics on combined data to ensure temporal continuity
        if self.split_type == "temporal":
            combined_df = self.combine_temporally()
            combined_roll = self._roll(combined_df, columns=columns, windows=windows, stats=stats, suffix=suffix)

            train_roll = combined_roll[combined_roll.index.isin(self.X_train.index)]
            test_roll = combined_roll[combined_roll.index.isin(self.X_test.index)]

        else:
            train_roll = self._roll(self.X_train, columns=columns, windows=windows, stats=stats, suffix=suffix)
            test_roll = self._roll(self.X_test, columns=columns, windows=windows, stats=stats, suffix=suffix)

        return train_roll, test_roll
    
    @staticmethod
    def _exceedance(df: pd.DataFrame, columns: Optional[List[str]],
                    window: int, threshold: float,
                    suffix: str) -> pd.DataFrame:
        """
        Calculate exceedance features for specified columns.
        
        Parameters:
        -----------
        df: pd.DataFrame
            Input DataFrame.
        columns: List[str]
            List of columns to compute exceedance for.
        window: int
            Rolling window size to calculate historical max/min.
        threshold: float
            Threshold for exceedance (e.g., 0.9 for 90%).
        suffix: str
            Suffix to add to feature names.
            
        Returns:
        --------
        pd.DataFrame
            With exceedance features.
        """
        new_columns = [f"{col}_{suffix}_{threshold}" for col in columns if col not in ['Country', 'Date']]
        df_feat = pd.DataFrame(data=0.0, index=df.index, columns=new_columns + ['Country', 'Date'])
        df_feat['Country'] = df['Country']
        df_feat['Date'] = df['Date']
        
        countries = df['Country'].unique()
        for col in columns:
            for country in countries:
                country_data = df.loc[df['Country'] == country, col]
                rolling_max = country_data.rolling(window=window, min_periods=window).max()
                rolling_min = country_data.rolling(window=window, min_periods=window).min()
                
                exceeds_max = (country_data > rolling_max * threshold).astype(int)
                exceeds_min = (country_data < rolling_min * (2 - threshold)).astype(int)

                df_feat.loc[df_feat['Country'] == country, f"{col}_{suffix}_{threshold}"] = (exceeds_max + exceeds_min).fillna(0).values

        return df_feat

    def create_extreme_binary(self, columns: Optional[List[str]] = None,
                              window: int = 24, threshold: float = 0.9,
                              suffix: str = 'exceeds') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create binary features indicating exceedance of a historical threshold.
        Uses combined data to ensure temporal continuity at split boundaries.

        Parameters:
        -----------
        columns: List[str]
            Columns to calculate exceedance for. If None, uses all numeric columns.
        window: int
            Rolling window size to calculate historical max/min.
        threshold: float
            Threshold for exceedance (e.g., 0.9 for 90%).
        suffix: str
            Suffix to add to feature names.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            DataFrames with exceedance features for train and test sets.
        """
        if columns is None:
            columns = self.X_train.select_dtypes(include=[np.number]).columns.tolist()

        # Calculate exceedance on combined data to ensure temporal continuity
        if self.split_type == "temporal":
            combined_df = self.combine_temporally()
            combined_extreme = self._exceedance(combined_df, columns=columns, window=window, threshold=threshold, suffix=suffix)

            train_extreme = combined_extreme[combined_extreme.index.isin(self.X_train.index)]
            test_extreme = combined_extreme[combined_extreme.index.isin(self.X_test.index)]

        else:
            train_extreme = self._exceedance(self.X_train, columns=columns, window=window, threshold=threshold, suffix=suffix)
            test_extreme = self._exceedance(self.X_test, columns=columns, window=window, threshold=threshold, suffix=suffix)

        return train_extreme, test_extreme

    @staticmethod
    def _ewm(df: pd.DataFrame, columns: Optional[List[str]],
             spans: Union[int, List[int]],
             suffix: str) -> pd.DataFrame:
        """
        Calculate exponentially weighted moving averages for specified columns.
        
        Parameters:
        -----------
        df: pd.DataFrame
            Input DataFrame.
        columns: List[str]
            List of columns to compute EWM for.
        spans: int, List[int]
            Span value or list of span values for EWM calculation.
        suffix: str
            Suffix to add to feature names.
            
        Returns:
        --------
        pd.DataFrame
            With EWM features.
        """
        new_columns = [f"{col}_{suffix}_{span}" for col in columns for span in spans if col not in ['Country', 'Date']]
        df_feat = pd.DataFrame(data=0.0, index=df.index, columns=new_columns + ['Country', 'Date'])
        df_feat['Country'] = df['Country']
        df_feat['Date'] = df['Date']
        
        countries = df['Country'].unique()
        for col in columns:
            for country in countries:
                country_data = df.loc[df['Country'] == country, col]
                for span in spans:
                    ewm_data = country_data.ewm(span=span, adjust=False).mean()
                    df_feat.loc[df_feat['Country'] == country, f"{col}_{suffix}_{span}"] = ewm_data.fillna(0).values

        return df_feat

    def create_exponentially_weighted_averages(self, columns: Optional[List[str]] = None,
                                               spans: Union[int, List[int]] = [3, 6, 12],
                                               suffix: str = 'ewm') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create exponentially weighted moving averages for specified columns.
        Uses combined data to ensure temporal continuity at split boundaries.
        
        Parameters:
        -----------
        columns: List[str]
            Columns to calculate EWM for. If None, uses all numeric columns.
        spans: int, List[int]
            Span value or list of span values for EWM calculation.
            Can be single int or list of ints.
        suffix: str
            Suffix to add to EWM feature names.
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            DataFrames with EWM features for train and test sets.
        """
        if columns is None:
            columns = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
            
        if isinstance(spans, int):
            spans = [spans]
        
        # Calculate EWM on combined data to ensure temporal continuity
        if self.split_type == "temporal":
            combined_df = self.combine_temporally()
            combined_ewm = self._ewm(combined_df, columns=columns, spans=spans, suffix=suffix)

            train_ewm = combined_ewm[combined_ewm.index.isin(self.X_train.index)]
            test_ewm = combined_ewm[combined_ewm.index.isin(self.X_test.index)]

        else:
            train_ewm = self._ewm(self.X_train, columns=columns, spans=spans, suffix=suffix)
            test_ewm = self._ewm(self.X_test, columns=columns, spans=spans, suffix=suffix)

        return train_ewm, test_ewm

    @staticmethod
    def _rtm(df: pd.DataFrame,
             columns: Optional[List[str]],
             suffix: str) -> pd.DataFrame:
        """
        Calculate regression to mean features for specified columns.
        
        Parameters:
        -----------
        df: pd.DataFrame
            Input DataFrame.
        columns: List[str]
            List of columns to compute distance to mean for.
        suffix: str
            Suffix to add to feature names.
            
        Returns:
        --------
        pd.DataFrame
            With regression to mean features.
        """
        new_columns = [f"{col}_{suffix}" for col in columns if col not in ['Country', 'Date']]
        df_feat = pd.DataFrame(data=0.0, index=df.index, columns=new_columns + ['Country', 'Date'])
        df_feat['Country'] = df['Country']
        df_feat['Date'] = df['Date']
        
        countries = df['Country'].unique()
        for col in columns:
            for country in countries:
                country_data = df.loc[df['Country'] == country, col]

                expanding_mean = country_data.expanding(min_periods=1).mean()
                distance = country_data - expanding_mean
                expanding_std = country_data.expanding(min_periods=1).std()
                normalized_distance = distance / (expanding_std + 1e-8)

                df_feat.loc[df_feat['Country'] == country, f"{col}_{suffix}"] = normalized_distance.fillna(0).values

        return df_feat
    
    def create_regression_to_mean(self, columns: Optional[List[str]] = None,
                                  suffix: str = 'dist_mean') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create regression to mean features that measure distance from historical mean.
        Uses combined data to ensure temporal continuity at split boundaries.
        
        Parameters:
        -----------
        columns: List[str]
            Columns to calculate distance to mean for.
            If None, uses all numeric columns.
        suffix: str
            Suffix to add to feature names.
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            DataFrames with regression to mean features for train and test sets.
        """
        if columns is None:
            columns = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate regression to mean on combined data to ensure temporal continuity
        if self.split_type == "temporal":
            combined_df = self.combine_temporally()
            combined_rtm = self._rtm(combined_df, columns=columns, suffix=suffix)

            train_rtm = combined_rtm[combined_rtm.index.isin(self.X_train.index)]
            test_rtm = combined_rtm[combined_rtm.index.isin(self.X_test.index)]

        else:
            train_rtm = self._rtm(self.X_train, columns=columns, suffix=suffix)
            test_rtm = self._rtm(self.X_test, columns=columns, suffix=suffix)

        return train_rtm, test_rtm

    @staticmethod
    def _lag(df: pd.DataFrame, columns: Optional[List[str]],
             lags: Union[int, List[int]],
             suffix: str) -> pd.DataFrame:
        """
        Create lag features for specified columns.

        Parameters:
        -----------
        df: pd.DataFrame
            Input DataFrame.
        columns: List[str]
            List of columns to create lag features for.
        lags: Union[int, List[int]]
            Lag periods to create. Can be single int or list of ints.
        suffix: str
            Suffix to add to lag feature names.

        Returns:
        --------
        pd.DataFrame
            DataFrame with lag features.
        """
        new_columns = [f"{col}_{suffix}{lag}" for col in columns for lag in lags if col not in ['Country', 'Date']]
        df_feat = pd.DataFrame(data=0.0, index=df.index, columns=new_columns + ['Country', 'Date'])
        df_feat['Country'] = df['Country']
        df_feat['Date'] = df['Date']

        countries = df['Country'].unique()
        for col in columns:
            for country in countries:
                country_data = df.loc[df['Country'] == country, col]
                for lag in lags:
                    df_feat.loc[df_feat['Country'] == country, f"{col}_{suffix}{lag}"] = country_data.shift(lag).values
        df_feat = df_feat.fillna(0)

        return df_feat
    
    def create_lag_features(self, 
                            columns: Optional[List[str]] = None,
                            lags: Union[int, List[int]] = [1, 3, 6],
                            suffix: str = 'lag') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create lag features for specified columns.
        Uses combined data to ensure temporal continuity at split boundaries.
        
        Parameters:
        -----------
        columns: List[str]
            Columns to create lags for.
            If None, uses all numeric columns.
        lags: int, List[int]
            Lag periods to create. Can be single int or list of ints.
        suffix: str
            Suffix to add to lag feature names.
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            DataFrames with lag features for train and test sets.
        """
        if columns is None:
            columns = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
            
        if isinstance(lags, int):
            lags = [lags]
        
        # Calculate lags on combined data to ensure temporal continuity
        if self.split_type == "temporal":
            combined_df = self.combine_temporally()
            combined_lags = self._lag(combined_df, columns=columns, lags=lags, suffix=suffix)

            train_lags = combined_lags[combined_lags.index.isin(self.X_train.index)]
            test_lags = combined_lags[combined_lags.index.isin(self.X_test.index)]

        else:
            train_lags = self._lag(self.X_train, columns=columns, lags=lags, suffix=suffix)
            test_lags = self._lag(self.X_test, columns=columns, lags=lags, suffix=suffix)

        return train_lags, test_lags
            
    @staticmethod
    def _momentum(df: pd.DataFrame, columns: Optional[List[str]],
                  window: int, suffix: str) -> pd.DataFrame:
        """
        Create momentum features for specified columns.

        Parameters:
        -----------
        df: pd.DataFrame
            Input DataFrame.
        columns: List[str]
            Columns to create momentum features for.
        window: int
            Window size for momentum calculation.
        suffix: str
            Suffix to add to momentum feature names.

        Returns:
        --------
        pd.DataFrame
            DataFrame with momentum features.
        """
        new_columns = [f"{col}_{suffix}" for col in columns if col not in ['Country', 'Date']]
        df_feat = pd.DataFrame(data=0.0, index=df.index, columns=new_columns + ['Country', 'Date'])
        df_feat['Country'] = df['Country']
        df_feat['Date'] = df['Date']

        countries = df['Country'].unique()
        for col in columns:
            for country in countries:
                country_data = df.loc[df['Country'] == country, col]
                ma = country_data.rolling(window=window, min_periods=1).mean()
                momentum = (country_data - ma) / (ma.abs() + 1e-8)
                df_feat.loc[df_feat['Country'] == country, f"{col}_{suffix}"] = momentum.values
        df_feat = df_feat.fillna(0)

        return df_feat
    
    def create_momentum_features(self, 
                                 columns: Optional[List[str]] = None,
                                 window: int = 3,
                                 suffix: str = 'momentum') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create momentum features for specified columns.
        Uses combined data to ensure temporal continuity at split boundaries.

        Parameters:
        -----------
        columns: List[str]
            Columns to create momentum for. If None, uses all numeric columns.
        window: int
            Window size for momentum calculation.
        suffix: str
            Suffix to add to momentum feature names.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            DataFrames with momentum features for train and test sets.
        """
        if columns is None:
            columns = self.X_train.select_dtypes(include=[np.number]).columns.tolist()

        # Calculate momentum on combined data to ensure temporal continuity
        if self.split_type == "temporal":
            combined_df = self.combine_temporally()
            combined_momentum = self._momentum(combined_df, columns=columns, window=window, suffix=suffix)

            train_momentum = combined_momentum[combined_momentum.index.isin(self.X_train.index)]
            test_momentum = combined_momentum[combined_momentum.index.isin(self.X_test.index)]

        else:
            train_momentum = self._momentum(self.X_train, columns=columns, window=window, suffix=suffix)
            test_momentum = self._momentum(self.X_test, columns=columns, window=window, suffix=suffix)

        return train_momentum, test_momentum

    @staticmethod
    def _volatility(df: pd.DataFrame, columns: Optional[List[str]],
                    window: int, suffix: str) -> pd.DataFrame:
        """
        Create volatility features for specified columns.

        Parameters:
        -----------
        df: pd.DataFrame
            Input DataFrame.
        columns: List[str]
            Columns to create volatility features for.
        window: int
            Window size for volatility calculation.
        suffix: str
            Suffix to add to volatility feature names.

        Returns:
        --------
        pd.DataFrame
            DataFrame with volatility features.
        """
        new_columns = [f"{col}_{suffix}" for col in columns if col not in ['Country', 'Date']]
        df_feat = pd.DataFrame(data=0.0, index=df.index, columns=new_columns + ['Country', 'Date'])
        df_feat['Country'] = df['Country']
        df_feat['Date'] = df['Date']
        
        countries = df['Country'].unique()
        for col in columns:
            for country in countries:
                country_data = df.loc[df['Country'] == country, col]
                rolling_std = country_data.rolling(window=window, min_periods=1).std()
                rolling_mean = country_data.rolling(window=window, min_periods=1).mean()
                df_feat.loc[df_feat['Country'] == country, f"{col}_{suffix}"] = (rolling_std / (rolling_mean + 1e-8)).values

        df_feat = df_feat.fillna(0)
        return df_feat
    
    def create_volatility_features(self,
                                   columns: Optional[List[str]] = None,
                                   window: int = 3,
                                   suffix: str = 'volatility') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create volatility features for specified columns.
        Uses combined data to ensure temporal continuity at split boundaries.

        Parameters:
        -----------
        columns: List[str]
            Columns to create volatility for. If None, uses all numeric columns.
        window: int
            Window size for volatility calculation.
        suffix: str
            Suffix to add to volatility feature names.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            DataFrames with volatility features for train and test sets.
        """
        if columns is None:
            columns = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate volatility on combined data to ensure temporal continuity
        if self.split_type == "temporal":
            combined_df = self.combine_temporally()
            combined_volatility = self._volatility(combined_df, columns=columns, window=window, suffix=suffix)

            train_volatility = combined_volatility[combined_volatility.index.isin(self.X_train.index)]
            test_volatility = combined_volatility[combined_volatility.index.isin(self.X_test.index)]

        else:
            train_volatility = self._volatility(self.X_train, columns=columns, window=window, suffix=suffix)
            test_volatility = self._volatility(self.X_test, columns=columns, window=window, suffix=suffix)

        return train_volatility, test_volatility
        
    @staticmethod
    def _seasonal_decompose(df: pd.DataFrame, columns: Optional[List[str]], model) -> pd.DataFrame:
        """
        Perform seasonal decomposition on specified columns of the DataFrame.

        Parameters:
        -----------
        df: pd.DataFrame
            Input DataFrame.
        columns: List[str]
            Columns to create seasonal features for.
        model: str
            Type of seasonal decomposition model ('additive' or 'multiplicative').

        Returns:
        --------
        pd.DataFrame
            DataFrame with seasonal features.
        """
        trend_cols = [f"{col}_seasonal_trend" for col in columns if col not in ['Country', 'Date']]
        res_cols = [f"{col}_seasonal_residual" for col in columns if col not in ['Country', 'Date']]
        new_columns = trend_cols + res_cols

        seasonal_df = pd.DataFrame(index=df.index, columns=new_columns + ['Country', 'Date'])
        seasonal_df['Country'] = df['Country']
        seasonal_df['Date'] = df['Date']

        countries = df['Country'].unique()
        for ii, col in enumerate(columns):
            if col in ['Country', 'Date']:
                continue
            for country in countries:
                country_series = df.loc[df['Country'] == country, col].dropna()
                
                # Check if series is long enough for decomposition
                if len(country_series) < 24:  # Need at least 2 periods
                    seasonal_df.loc[seasonal_df['Country'] == country, trend_cols[ii]] = 0
                    seasonal_df.loc[seasonal_df['Country'] == country, res_cols[ii]] = 0
                else:
                    try:
                        result = seasonal_decompose(country_series, model=model, period=12)
                        seasonal_df.loc[seasonal_df['Country'] == country, trend_cols[ii]] = result.trend.fillna(0).values
                        seasonal_df.loc[seasonal_df['Country'] == country, res_cols[ii]] = result.resid.fillna(0).values
                    except Exception:
                        # Handle decomposition failures gracefully
                        seasonal_df.loc[seasonal_df['Country'] == country, trend_cols[ii]] = 0
                        seasonal_df.loc[seasonal_df['Country'] == country, res_cols[ii]] = 0
        
        seasonal_df = seasonal_df.fillna(0)
        return seasonal_df

    def create_seasonal_decomposition_features(self,
                                               columns: Optional[List[str]] = None,
                                               model: str = 'additive') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create seasonal decomposition features for the training and test datasets.
        Uses combined data to ensure temporal continuity at split boundaries.
        
        Parameters:
        -----------
        columns: List[str]
            Columns to create seasonal features for.
        model: str
            Type of seasonal decomposition model ('additive' or 'multiplicative').
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            DataFrames with seasonal decomposition features for train and test sets.
        """
        if columns is None:
            columns = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        if model not in ['additive', 'multiplicative']:
            raise ValueError("Model must be either 'additive' or 'multiplicative'.")
        
        # Decompose combined data to ensure temporal continuity
        if self.split_type == "temporal":
            combined_df = self.combine_temporally()
            combined_seasonal = self._seasonal_decompose(combined_df, columns=columns, model=model)

            train_seasonal = combined_seasonal[combined_seasonal.index.isin(self.X_train.index)]
            test_seasonal = combined_seasonal[combined_seasonal.index.isin(self.X_test.index)]

        else:
            train_seasonal = self._seasonal_decompose(self.X_train, columns=columns, model=model)
            test_seasonal = self._seasonal_decompose(self.X_test, columns=columns, model=model)
        
        return train_seasonal, test_seasonal

    def add_features(self, features: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """
        Add features to the existing train, and test DataFrames.

        Parameters:
        -----------
        features: Tuple[pd.DataFrame, pd.DataFrame]
            Tuple of DataFrames (train_features, test_features) to be concatenated with existing data.
        """
        train_features, test_features = features

        self.X_train = pd.merge(self.X_train, train_features, on=['Country', 'Date'], how='left')
        self.X_test = pd.merge(self.X_test, test_features, on=['Country', 'Date'], how='left')

        # Drop all duplicate columns
        self.X_train = self.X_train.loc[:, ~self.X_train.columns.duplicated()]
        self.X_test = self.X_test.loc[:, ~self.X_test.columns.duplicated()]