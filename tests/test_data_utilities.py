from src.data.data_utilities import get_series_frequency, join_timelines, merge_timeseries

import pandas as pd

def test_get_series_frequency():
    # Test with daily dates
    daily_dates = pd.Series(pd.date_range(start='2023-01-01', periods=10, freq='D'))
    assert get_series_frequency(daily_dates) == "D"

    # Test with monthly dates
    monthly_dates = pd.Series(pd.date_range(start='2023-01-01', periods=10, freq='M'))
    assert get_series_frequency(monthly_dates) == "M"

    # Test with quarterly dates
    quarterly_dates = pd.Series(pd.date_range(start='2023-01-01', periods=10, freq='Q'))
    assert get_series_frequency(quarterly_dates) == "Q"

    # Test with yearly dates
    yearly_dates = pd.Series(pd.date_range(start='2023-01-01', periods=10, freq='Y'))
    assert get_series_frequency(yearly_dates) == "Y"

    # Test with insufficient data
    insufficient_dates = pd.Series(pd.date_range(start='2023-01-01', periods=1))
    assert get_series_frequency(insufficient_dates) == "Unknown"

    # Test with mixed frequency dates
    mixed_dates = pd.Series([pd.Timestamp('2023-01-01'), pd.Timestamp('2023-02-01'), pd.Timestamp('2023-03-15')])
    assert get_series_frequency(mixed_dates) == "Unknown"

    # Test with missing vallues
    missing_daily = daily_dates.drop([3,7]).reset_index(drop=True)
    assert get_series_frequency(missing_daily) == "D"

def test_join_timelines():
    # Test with two daily timelines
    dates1 = pd.Series(pd.date_range(start='2023-01-01', periods=10, freq='D'))
    dates2 = pd.Series(pd.date_range(start='2023-01-03', periods=10, freq='D'))
    joint_timeline = join_timelines(dates1, dates2)
    assert joint_timeline.freq == 'D'
    assert len(joint_timeline) == 12  # 12 days from 2023-01-01 to 2023-01-12

    # Test with two monthly timelines
    dates1 = pd.Series(pd.date_range(start='2023-01-01', periods=3, freq='M'))
    dates2 = pd.Series(pd.date_range(start='2023-02-01', periods=3, freq='M'))
    joint_timeline = join_timelines(dates1, dates2)
    assert joint_timeline.freq == 'M'
    assert len(joint_timeline) == 4  # 4 months from Jan to Apr

    # Test with empty series
    empty_dates1 = pd.Series(dtype='datetime64[ns]')
    empty_dates2 = pd.Series(dtype='datetime64[ns]')
    joint_timeline = join_timelines(empty_dates1, empty_dates2)
    assert len(joint_timeline) == 0

    # Test with mixed frequencies
    dates1 = pd.Series(pd.date_range(start='2023-01-01', periods=15, freq='D'))
    dates2 = pd.Series(pd.date_range(start='2023-01-01', periods=2, freq='M'))
    joint_timeline = join_timelines(dates1, dates2)
    assert joint_timeline.freq == 'D'  # Daily is the highest frequency

    # Test with one empty series
    dates1 = pd.Series(pd.date_range(start='2023-01-01', periods=5, freq='D'))
    empty_dates2 = pd.Series(dtype='datetime64[ns]')
    joint_timeline = join_timelines(dates1, empty_dates2)
    assert pd.Series(joint_timeline).equals(dates1)  # Should return the first series as the joint timeline

    # Test with invalid input types
    try:
        join_timelines("not a series", dates2)
    except ValueError as e:
        assert str(e) == "Both dates1 and dates2 must be pandas Series"
    try:
        join_timelines(dates1, "not a series")
    except ValueError as e:
        assert str(e) == "Both dates1 and dates2 must be pandas Series"

def test_merge_timeseries():
    # Test merging three dataframes with overlapping dates
    df1 = pd.DataFrame({'Date': pd.date_range(start='2023-01-01', periods=5, freq='D'), 'Col1': range(5)})
    df2 = pd.DataFrame({'Date': pd.date_range(start='2023-01-03', periods=5, freq='D'), 'Col2': range(5, 10)})
    df3 = pd.DataFrame({'Date': pd.date_range(start='2023-01-04', periods=5, freq='D'), 'Col3': range(10, 15)})
    df1['Country'], df2['Country'] = 'A', 'B'
    df1.loc[2, 'Country'] = 'B'
    df2.loc[3:, 'Country'] = 'A'
    dfs = [df1, df2, df3]

    ## Test merge on 'Country' column
    merged_df = merge_timeseries(dfs, on='Country')
    assert merged_df.index.names == ['Country', 'Date']
    assert 'Col1' in merged_df.columns
    assert 'Col2' in merged_df.columns
    assert 'Col3' in merged_df.columns
    assert len(merged_df) == 16  # Should include all unique date x country combinations
    assert merged_df.isna().sum().sum() == 28
    # {'Col1': 11, 'Col2': 11, 'Col3': 6}  # Check for NaNs in merged columns

    merged_df = merge_timeseries(dfs, on='')
    assert 'Col1' in merged_df.columns
    assert 'Col2' in merged_df.columns
    assert 'Col3' in merged_df.columns
    assert 'Country' in merged_df.columns
    assert len(merged_df) == 8  # Should include all unique dates
    assert merged_df.isna().sum().sum() == 12
    # {'Col1': 3, 'Col2': 3, 'Col3': 3, 'Country': 3}  # No NaNs in merged columns