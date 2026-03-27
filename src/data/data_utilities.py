import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login
from typing import List, Union, Dict, Tuple
import os

def upload_to_huggingface(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                          repo_name: str,
                          config_name: str = None) -> None:
    """
    Upload dataset to Hugging Face Hub with config_name as the subset name.
    
    Parameters:
    -----------
    data: pd.DataFrame or Dictionary of DataFrames
        to upload to HuggingFace Hub.
    repo_name: str
        Repository name.
    config_name: str
        Configuration name (subset name). If None, uses default naming.

    Raises:
    -------
    ValueError: If required environment variables are not set.
    Exception: If upload to HuggingFace fails.
    """
    # Login to Hugging Face
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN not set in .env")

    login(token=token, new_session=False,
            add_to_git_credential=True)

    try:
        if isinstance(data, pd.DataFrame):
            # Replace 'NA' and similar string values with np.nan
            data = data.replace(['NA', 'N/A', '', 'nan'], np.nan)

            dataset = Dataset.from_pandas(data)
            dataset.push_to_hub(repo_name, config_name)

        elif isinstance(data, dict):
            for key, df in data.items():
                # Replace 'NA' and similar string values with np.nan
                df = df.replace(['NA', 'N/A', '', 'nan'], np.nan)
                data[key] = Dataset.from_pandas(df)
                
            dataset_dict = DatasetDict(data)
            dataset_dict.push_to_hub(repo_name, config_name)

    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")

        if isinstance(data, pd.DataFrame):
            data = data.replace(['NA', 'N/A', '', 'nan'], np.nan)
            data.to_parquet(f'data/{repo_name}/{config_name}.parquet')

        elif isinstance(data, dict):
            for key, df in data.items():
                df = df.replace(['NA', 'N/A', '', 'nan'], np.nan)
                df.to_parquet(f'data/{repo_name}/{config_name}/{key}.parquet')

def get_series_frequency(dates: pd.Series) -> str:
    """
    Determine the frequency of a time series based on date differences.
    
    Parameters:
    -----------
    dates: pd.Series
        A pandas Series containing datetime values.
        
    Returns:
    --------
    str
        A string indicating the frequency: 'D' for daily, 'M' for monthly, 'Q' for quarterly, 'Y' for yearly, or a custom format for other intervals. Returns 'unknown' if insufficient data is provided.
        
    Raises:
    -------
    ValueError: If dates parameter is not a pandas Series.
    """
    if not isinstance(dates, pd.Series):
        raise ValueError("dates must be a pandas Series")
    
    if len(dates) < 2:
        return "Unknown"

    # Convert to datetime and calculate differences
    dates = pd.to_datetime(dates).sort_values()
    diffs = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
    mode_diff = max(set(diffs), key=diffs.count)

    # Determine frequency based on mode difference
    if mode_diff == 1:
        return "D"
    elif 28 <= mode_diff <= 31:
        return "M"
    elif 89 <= mode_diff <= 92:
        return "Q"
    elif 365 <= mode_diff <= 366:
        return "Y"
    return f"Unknown"

def join_timelines(dates1: pd.Series, dates2: pd.Series = None) -> pd.DatetimeIndex:
    """
    Create the highest frequency joint timeline from two date frequencies.
    
    Parameters:
    -----------
    dates1: pd.Series
        A pandas Series containing datetime values from first timeline.
    dates2: pd.Series
        A pandas Series containing datetime values from second timeline.
        
    Returns:
    --------
    pd.DatetimeIndex
        A pandas DatetimeIndex representing the combined timeline with appropriate frequency.
        
    Raises:
    -------
    ValueError: If either dates1 or dates2 is not a pandas Series.
    TypeError: If dates cannot be converted to datetime format.
    """
    if dates2 is None:
        dates2 = pd.Series(dtype='datetime64[ns]')
    
    if not isinstance(dates1, pd.Series) or not isinstance(dates2, pd.Series):
        raise ValueError("Both dates1 and dates2 must be pandas Series")
    
    try:
        if len(dates1) == 0 and len(dates2) == 0:
            return pd.DatetimeIndex([])

        # Convert to datetime and get min and max dates
        if len(dates1) > 0:
            dates1_dt = pd.to_datetime(dates1)
            mindate1, maxdate1 = dates1_dt.min().date(), dates1_dt.max().date()
        if len(dates2) > 0:
            dates2_dt = pd.to_datetime(dates2)
            mindate2, maxdate2 = dates2_dt.min().date(), dates2_dt.max().date()
            
    except (AttributeError, ValueError):
        raise TypeError("dates1 and dates2 must contain datetime-convertible values")
    
    # Determine date range
    if len(dates1) > 0 and len(dates2) > 0:
        mindate = min(mindate1, mindate2)
        maxdate = max(maxdate1, maxdate2)
    elif len(dates1) > 0:
        mindate, maxdate = mindate1, maxdate1
    elif len(dates2) > 0:
        mindate, maxdate = mindate2, maxdate2
    else:
        return pd.DatetimeIndex([])
    
    # Determine frequency using original series
    freq_importance = {'D': 4, 'M': 3, 'Q': 2, 'Y': 1, 'Unknown': 0}
    freq1 = get_series_frequency(dates1) if len(dates1) > 1 else 'Unknown'
    freq2 = get_series_frequency(dates2) if len(dates2) > 1 else 'Unknown'

    freq = freq1 if freq_importance[freq1] > freq_importance[freq2] else freq2

    # if freq != "D":
    #     freq += "E"  # end of month/year (to ensure no look ahead bias)
    
    # Return DatetimeIndex
    return pd.date_range(start=mindate, end=maxdate, freq=freq)

def merge_timeseries(dfs: List[pd.DataFrame], on: str = 'Country') -> pd.DataFrame:
    """
    Memory-efficient merge of multiple dataframes with a common date index.
    
    Parameters:
    -----------
    dfs: List of DataFrames
        List of DataFrames to merge.
    on: str
        Column to merge on. Defaults to 'Country'.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing all columns from the input dataframes, indexed by Country and Date.

    Raises:
    -------
    ValueError: If dfs is empty or if any dataframe does not contain the 'Date' column.
    """
    if not dfs or not isinstance(dfs, list):
        raise ValueError("dfs must be a non-empty list of DataFrames")

    for df in dfs:
        if 'Date' not in df.columns:
            raise ValueError("Each DataFrame must contain a 'Date' column")

    all_dates_set = set()
    unique_values = set()
    
    for df in dfs:
        all_dates_set.update(pd.to_datetime(df['Date']).dt.date)
        if on and on in df.columns:
            unique_values.update(df[on].unique())
    
    # Create sorted timeline
    all_dates = pd.DatetimeIndex(sorted(all_dates_set))
    use_multiindex = bool(on and unique_values)
    processed_dfs = []
    
    for df in dfs:
        df_copy = df.copy()
        
        if use_multiindex:
            if on in df_copy.columns:
                df_copy = df_copy.set_index([on, 'Date'])
            else:
                # Broadcast to all unique values
                expanded_rows = []
                for value in sorted(unique_values):
                    temp_df = df_copy.copy()
                    temp_df[on] = value
                    expanded_rows.append(temp_df)
                df_copy = pd.concat(expanded_rows, ignore_index=True)
                df_copy = df_copy.set_index([on, 'Date'])
        else:
            df_copy = df_copy.set_index('Date')
        
        processed_dfs.append(df_copy)
    
    merged_df = processed_dfs[0]
    
    for df_to_merge in processed_dfs[1:]:
        merged_df = pd.concat([merged_df, df_to_merge], axis=1, sort=False)
        # Remove duplicate columns
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    
    # Re-index
    if use_multiindex:
        new_index = pd.MultiIndex.from_product(
            [sorted(unique_values), all_dates], 
            names=[on, 'Date']
        )
        merged_df = merged_df.reindex(new_index)
    else:
        merged_df = merged_df.reindex(all_dates)
    
    return merged_df


def build_labels(df: pd.DataFrame) -> np.array:
    """
    Build crisis labels for the DataFrame based on a master crisis labels file.

    Parameters:
    -----------
    df: pd.DataFrame
        DataFrame containing country-date panel data.

    Returns:
    --------
    np.array
        Array of crisis labels (0 or 1) for each row in the DataFrame.
    """
    if 'Country' not in df.columns or 'Date' not in df.columns:
        raise ValueError("Dataframe must contain 'Country' and 'Date' columns")

    username = os.getenv("HUGGINGFACE_USERNAME")
    if not username:
        raise ValueError("HUGGINGFACE_USERNAME not set in .env")
    
    # Load crisis labels
    datalink = f"{username}/crisis-labels-dataset"
    crisis_labels = load_dataset(datalink)['train'].to_pandas()
    
    y = np.zeros(len(df), dtype=int)

    # Map years to crisis labels
    countries = df['Country']
    years = pd.to_datetime(df['Date']).dt.year
    for i, (country, year) in enumerate(zip(countries, years)):
        crisis_years = crisis_labels.loc[crisis_labels['Country'] == country, 'Year'].values
        if year in crisis_years:
            y[i] = 1

    return y

def drop_recovery(df: pd.DataFrame, y: np.array, recovery_years: int = 4) -> Tuple[pd.DataFrame, np.array]:
    """
    Drop recovery years from the DataFrame based on crisis labels.

    Parameters:
    -----------
    df: pd.DataFrame
        DataFrame containing country-date panel data.
    y: np.array
        Array of crisis labels (0 or 1) for each row in the DataFrame.
    recovery_years: int
        Number of years to drop after a crisis label.

    Returns:
    --------
    Tuple[pd.DataFrame, np.array]
        A tuple containing the filtered DataFrame and the updated crisis labels array.
    """
    if 'Country' not in df.columns or 'Date' not in df.columns:
        raise ValueError("Dataframe must contain 'Country' and 'Date' columns")
    
    df_copy = df.copy()
    y_copy = y.copy()
    
    # Convert Date to datetime
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    
    # Create a mask for dropping recovery years
    drop_mask = np.zeros(len(y_copy), dtype=bool)
    
    for i, (country, date) in enumerate(zip(df_copy['Country'], df_copy['Date'])):
        if y_copy[i] == 1:  # If crisis label is positive
            # Calculate the end date for dropping recovery years
            recovery_end_date = date + pd.DateOffset(years=recovery_years)
            
            # Find the next positive label for the same country
            country_mask = (df_copy['Country'] == country) & (df_copy['Date'] > date)
            future_indices = df_copy[country_mask].index
            
            next_crisis_date = None
            for future_idx in future_indices:
                if y_copy[future_idx] == 1:
                    next_crisis_date = df_copy.loc[future_idx, 'Date']
                    break
            
            # Determine the actual end date (whichever is shorter)
            if next_crisis_date is not None:
                actual_end_date = min(recovery_end_date, next_crisis_date)
            else:
                actual_end_date = recovery_end_date
            
            # Create mask for dates to drop (excluding the next crisis date itself)
            mask = (df_copy['Country'] == country) & (df_copy['Date'] > date) & (df_copy['Date'] < actual_end_date)
            drop_mask[mask] = True
    
    # Drop rows where drop_mask is True
    df_filtered = df_copy[~drop_mask].reset_index(drop=True)
    y_filtered = y_copy[~drop_mask]

    return df_filtered, y_filtered