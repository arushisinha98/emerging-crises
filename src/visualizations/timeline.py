import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.patches import Patch, Rectangle
import pandas as pd
from datasets import load_dataset

import os
import dotenv
dotenv.load_dotenv()
username = os.getenv("HUGGINGFACE_USERNAME")

from .utilities import load_crisis_data, get_universal_region_colors

_UNIVERSAL_REGION_COLORS = None

def plot_prediction_timeline(model, data_df, subset, color_by='region', 
                             title=None, date_column='Date', country_column='Country',
                             figsize=(12, 10), bar_height=1.2,
                             threshold=0.5):
    """
    Plot model predictions vs actual crisis labels in a timeline format.
    Shows actual crisis periods as filled bars and predictions as X marks.
    Works with both sklearn and PyTorch models.
    
    Parameters:
    -----------
    model : sklearn estimator or PyTorch model
        Trained model that can make predictions
    data_df : pd.DataFrame
        DataFrame containing the prediction data with columns: [country_column, date_column, features...]
    subset : str
        Either 'developed' or 'emerging' to load the appropriate crisis data
    color_by : str
        Column name to use for region-based coloring (default: 'region')
    title : str
        Plot title
    date_column : str
        Name of the date column in data_df (default: 'Date')
    country_column : str
        Name of the country column (default: 'Country')
    figsize : tuple
        Figure size (default: (12, 10))
    bar_height : float
        Height of the country bars (default: 1.2)
    threshold : float
        Prediction threshold for binary classification (default: 0.5)
        Only used for sklearn models with predict_proba capability
    """
    global _UNIVERSAL_REGION_COLORS
    
    # Initialize universal colors if not already done
    if _UNIVERSAL_REGION_COLORS is None:
        _UNIVERSAL_REGION_COLORS = get_universal_region_colors()
    
    # Load crisis data
    crisis_df = load_crisis_data(subset=subset)
    # Check all columns exist in dataframe
    assert color_by in crisis_df.columns, f"Color by '{color_by}' must be in crisis_df"
    assert country_column in data_df.columns and country_column in crisis_df.columns, f"Country column '{country_column}' must be in both DataFrames"
    assert date_column in data_df.columns, f"Date column '{date_column}' must be in data_df"
    
    # Convert date column to datetime
    if not pd.api.types.is_datetime64_any_dtype(data_df[date_column]):
        data_df = data_df.copy()
        data_df[date_column] = pd.to_datetime(data_df[date_column])
    
    # Get unique regions and countries, sorted by region then country
    regions = crisis_df[color_by].unique()
    countries_df = crisis_df.groupby([color_by, country_column]).size().reset_index()
    countries_df = countries_df.sort_values([color_by, country_column])
    countries = countries_df[country_column].unique()
    
    # Filter to only countries that exist in both dataframes
    countries = [country for country in countries if country in data_df[country_column].values]
    
    # Use universal color mapping
    region_colors = {region: _UNIVERSAL_REGION_COLORS.get(region, 'gray') 
                    for region in regions}

    # Assign colors to countries based on their region using universal colors
    country_colors = {}
    for country in countries:
        if country in crisis_df[country_column].values:
            country_region = crisis_df[crisis_df[country_column] == country][color_by].iloc[0]
            country_colors[country] = region_colors[country_region]
        else:
            country_colors[country] = 'gray'
    
    min_date = data_df[date_column].min() - pd.Timedelta(days=180)
    max_date = data_df[date_column].max() + pd.Timedelta(days=180)
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(figsize[0], len(countries) * bar_height/4))
    legend_elements = []
    plotted_regions = set()

    feature_columns = [col for col in data_df.columns if col not in [country_column, date_column]]
    
    # Plot each country's crisis periods
    for i, country in enumerate(countries):
        country_crises = crisis_df[crisis_df[country_column] == country]
        country_data = data_df[data_df[country_column] == country].sort_values(date_column).copy()
        
        if len(country_data) == 0:
            continue
        
        # Get country's region and color using universal mapping
        if country in crisis_df[country_column].values:
            country_region = crisis_df[crisis_df[country_column] == country][color_by].iloc[0]
            color = country_colors.get(country, 'gray')
        else:
            country_region = 'Unknown'
            color = 'gray'
        
        # Plot crisis label periods as filled bars
        for _, crisis_row in country_crises.iterrows():
            crisis_year = crisis_row['Year']
            # Convert crisis year to approximate date range for the year
            crisis_start = pd.Timestamp(f'{crisis_year}-01-01')
            crisis_end = pd.Timestamp(f'{crisis_year}-12-31')
            
            # Only plot if crisis overlaps with prediction period
            if crisis_end >= min_date and crisis_start <= max_date:
                plot_start = max(crisis_start, min_date)
                plot_end = min(crisis_end, max_date)
                
                rect = Rectangle((plot_start, i - bar_height/2), 
                                  plot_end - plot_start, bar_height,
                                  facecolor=color, alpha=0.3, edgecolor=color, linewidth=1)
                ax.add_patch(rect)
        
        if len(feature_columns) > 0:
            # Check if predictions have been passed in data_df
            if 'y_true' in country_data.columns and 'y_pred' in country_data.columns:
                aligned_data = country_data.copy()
                aligned_data['prediction'] = aligned_data['y_pred']
            
            # Handle different model types
            elif hasattr(model, 'predict_with_indices') and hasattr(model, 'get_aligned_labels'):
                # LSTMClassifier case
                try:
                    predictions, _, prediction_indices = model.predict_with_indices(country_data)
                    
                    # Create aligned data using the prediction indices returned by the model
                    # The prediction_indices tell us which rows from the original data correspond to predictions
                    aligned_data = country_data.iloc[prediction_indices].copy()
                    
                    # Ensure lengths match
                    min_length = min(len(aligned_data), len(predictions))
                    aligned_data = aligned_data.iloc[:min_length].copy()
                    aligned_data['prediction'] = predictions[:min_length]
                    
                except Exception as e:
                    print(f"Warning: Could not get predictions for {country}: {e}")
                    continue
            
            elif hasattr(model, 'predict') and hasattr(model, 'predict_proba') and hasattr(model, 'fnn_model'):
                # FNNClassifier case
                predictions = model.predict(country_data)
                aligned_data = country_data.copy()
                aligned_data['prediction'] = predictions[:len(country_data)]
            
            else:
                # Standard sklearn model case
                X_country = country_data[feature_columns]
                
                # Use predict_proba with custom threshold if available
                if hasattr(model, 'predict_proba'):
                    try:
                        probabilities = model.predict_proba(X_country)
                        positive_probs = probabilities[:, 1]
                        predictions = (positive_probs >= threshold).astype(int)
                    except Exception:
                        predictions = model.predict(X_country)
                else:
                    predictions = model.predict(X_country)
                
                aligned_data = country_data.copy()
                predictions = predictions[:len(country_data)]
                aligned_data['prediction'] = predictions
            
            # Add X marks for positive predictions
            positive_predictions = aligned_data[aligned_data['prediction'] == 1]
            for _, pred_row in positive_predictions.iterrows():
                pred_date = pred_row[date_column]
                ax.plot(pred_date, i, 'x', color=color, markersize=4, markeredgewidth=1)
        
        # Add to legend if this region hasn't been added yet
        if country_region not in plotted_regions and country_region != 'Unknown':
            legend_elements.append(Patch(facecolor=color, alpha=0.3, edgecolor=color,
                                       label=f"Crisis labels in {country_region}"))
            plotted_regions.add(country_region)
    
    # Add legend elements for ALL regions (not just those in current plot)
    # This ensures consistency across all timeline plots
    all_regions_sorted = sorted(_UNIVERSAL_REGION_COLORS.keys())
    legend_elements = []  # Reset to include all regions
    
    for region in all_regions_sorted:
        # Check if region appears in current plot
        if region in regions:
            label = f"Crisis labels in {region}"
            alpha = 0.3
        else:
            label = f"{region} (not in {subset})"
            alpha = 0.15  # More transparent for regions not in current plot
        
        legend_elements.append(Patch(facecolor=_UNIVERSAL_REGION_COLORS[region], 
                                   alpha=alpha, edgecolor=_UNIVERSAL_REGION_COLORS[region],
                                   label=label))
    
    ax.set_xlim(min_date, max_date)
    ax.set_ylim(-0.5, len(countries) - 0.5)
    ax.set_xlabel('Year', fontsize=12)
    
    ax.xaxis.set_major_locator(mdates.YearLocator(month=7, day=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.YearLocator(month=12, day=31))

    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries)
    
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    
    region_labels = []
    seen_regions = set()
    
    for i, country in enumerate(countries):
        if country in crisis_df[country_column].values:
            country_region = crisis_df[crisis_df[country_column] == country][color_by].iloc[0]
            if country_region not in seen_regions:
                region_labels.append(country_region)
                seen_regions.add(country_region)
            else:
                region_labels.append('')
        else:
            region_labels.append('')
    
    ax2.set_yticks(range(len(countries)))
    ax2.set_yticklabels(region_labels)
    ax2.invert_yaxis()
    
    ax.grid(True, which='minor', axis='x', alpha=0.3, zorder=0.5)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    
    # Add X mark legend and finalize
    if legend_elements:
        legend_elements.append(plt.Line2D([0], [0], marker='x', color='black', linewidth=0,
                                        markersize=8, markeredgewidth=2, label='Model prediction'))
        
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(0, 100*(-0.05)/len(countries)), 
                 ncol=min(len(legend_elements), 4), frameon=False)
    
    plt.title(title if title else "Model Predictions vs True Crisis Labels", fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_ticker_vs_crises(ticker, country, title="", config_path='../src/config.json', figsize=(12, 4),
                         x_limits=None, highlights=None):
    """
    Plot ticker data vs crisis periods with optional x-axis limits and highlights.
    
    Parameters:
    -----------
    ticker : str
        Yahoo Finance ticker symbol
    country : str
        Country name to get crisis data for
    title : str
        Plot title
    config_path : str
        Path to config file
    figsize : tuple
        Figure size (width, height)
    x_limits : tuple of str, optional
        Tuple of (start_date, end_date) strings to set x-axis limits
        Can be one-sided: ('2007-02-28', None) or (None, '2023-12-31')
        Example: ('2004-06-30', '2023-12-31')
    highlights : list of tuples, optional
        List of (start_date, end_date) string tuples to highlight as cross-hatched regions
        Example: [('2022-09-01','2022-09-30'), ('2019-11-01','2020-06-30')]
    """
    # Get crises for country
    crises = load_dataset(f"{username}/crisis-labels-dataset", split='train').to_pandas()
    country_crises = crises[crises['Country'] == country].sort_values('Year')

    # Get ticker from Yahoo Finance
    from src.data.loader import YahooFinanceDataLoader
    loader = YahooFinanceDataLoader(config_path=config_path)
    ticker_data = loader.download_series_data(variables=[ticker])

    # Plot the ticker data
    plt.figure(figsize=figsize)
    plt.plot(ticker_data['Date'], ticker_data[f'{ticker}.Open'], label=f'{ticker} Open Price',
             color='black', linewidth=0.75)
    
    # Add crisis periods as shaded regions
    for i, (_, row) in enumerate(country_crises.iterrows()):
        plt.axvspan(pd.Timestamp(f"{row['Year']}-01-01"), pd.Timestamp(f"{row['Year']}-12-31"), 
                    color='red', alpha=0.3, label='Pre-Crisis Period' if i == 0 else "")
    
    # Add highlight periods as cross-hatched regions
    if highlights:
        for i, (start_date, end_date) in enumerate(highlights):
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            plt.axvspan(start_ts, end_ts, color='blue', alpha=0.3, hatch='///', 
                       label='Positive Prediction' if i == 0 else "")
    
    if title:
        plt.title(f"{title}")
    else:
        plt.title(f"{ticker} Price vs {country} Pre-Crisis Period")

    # Set x-axis limits
    if x_limits:
        # Handle one-sided or two-sided x_limits
        if x_limits[0] is not None and x_limits[1] is not None:
            # Both limits provided
            start_limit = pd.Timestamp(x_limits[0])
            end_limit = pd.Timestamp(x_limits[1])
        elif x_limits[0] is not None and x_limits[1] is None:
            # Only start limit provided
            start_limit = pd.Timestamp(x_limits[0])
            end_limit = ticker_data['Date'].max() + pd.Timedelta(days=30)
        elif x_limits[0] is None and x_limits[1] is not None:
            # Only end limit provided
            start_limit = ticker_data['Date'].min() - pd.Timedelta(days=30)
            end_limit = pd.Timestamp(x_limits[1])
        else:
            # Both are None, use default
            start_limit = ticker_data['Date'].min() - pd.Timedelta(days=30)
            end_limit = ticker_data['Date'].max() + pd.Timedelta(days=30)
        
        # Ensure limits are valid before setting
        if pd.isna(start_limit) or pd.isna(end_limit):
            print(f"Warning: Invalid date limits detected. Using default range.")
            start_limit = ticker_data['Date'].min() - pd.Timedelta(days=30)
            end_limit = ticker_data['Date'].max() + pd.Timedelta(days=30)
        
        plt.xlim(start_limit, end_limit)
        
        # Filter data within x_limits for y-axis calculation
        mask = (ticker_data['Date'] >= start_limit) & (ticker_data['Date'] <= end_limit)
        filtered_data = ticker_data[mask]
        
        if len(filtered_data) > 0:
            y_max = filtered_data[f'{ticker}.Open'].max()
            plt.ylim(0, y_max * 1.05)  # Add 5% padding on top
        else:
            # If no data in range, use full data range
            y_max = ticker_data[f'{ticker}.Open'].max()
            plt.ylim(0, y_max * 1.05)
    else:
        # Default behavior - use full range
        min_date = ticker_data['Date'].min() - pd.Timedelta(days=30)
        max_date = ticker_data['Date'].max() + pd.Timedelta(days=30)
        plt.xlim(min_date, max_date)
        
        # Set y-axis from 0 to max of all data
        y_max = ticker_data[f'{ticker}.Open'].max()
        plt.ylim(0, y_max * 1.05)  # Add 5% padding on top
    
    # Format x-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator(month=7, day=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.YearLocator(month=12, day=31))
    
    # Add grid lines at year boundaries
    ax.grid(True, which='minor', axis='x', alpha=0.3, zorder=0.5)
    ax.set_axisbelow(True)
    
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_variable_vs_crises(df, columns, countries, figsize=(8,6)):
    """
    Plot specified variables over time, highlighting crisis periods.

    Parameters:
    -----------
    df: pd.DataFrame
        The input DataFrame containing time series data.
    columns: List[str]
        The names of the columns to plot.
    countries: List[str]
        The names of the countries to include in the plot.
    figsize: Tuple[int, int]
        The size of the figure to create.
    """
    # Get crises for countries
    crises = load_dataset(f"{username}/crisis-labels-dataset", split='train').to_pandas().set_index('Country')

    fig, axes = plt.subplots(len(columns), 1, figsize=(figsize[0], figsize[1]*len(columns)))

    for country in countries:
        # Extract crisis years
        country_df = df[df['Country'] == country]
        crisis_years = crises.loc[country, 'Year'].values
        dates = country_df['Date'].dt.year
        crisis_idx = [i for i in dates.index if dates[i] in crisis_years]

        # Plot variable over time, scatter crisis
        for ii, col in enumerate(columns):
            axes[ii].plot(country_df['Date'], country_df[col], linewidth=2, label=country)
            axes[ii].scatter([country_df.loc[i,'Date'] for i in crisis_idx],
                             [country_df.loc[i,col] for i in crisis_idx],
                             s=15, marker='x', zorder=5, color='gray')
            axes[ii].set_title(f'{col}', fontsize=14)
            axes[ii].grid(True, alpha=0.3)
            if ii == 0:
                axes[ii].legend()