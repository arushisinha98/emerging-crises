# Timeline Visualization Documentation

The timeline visualization functions in `src/visualizations/timeline.py` handle
- Model prediction timeline visualization compared to actual crisis labels
- Stock ticker analysis against crisis periods
- Multi-variable time series visualization with crisis highlighting
- Interactive timeline plots for financial data analysis

The utilities are designed to work with both sklearn and PyTorch models, supporting comprehensive timeline-based analysis of financial crisis prediction models and market data.

## Usage Instructions

```python
from src.visualizations.timeline import (
    plot_prediction_timeline,
    plot_ticker_vs_crises,
    plot_variable_vs_crises
)

# Plot model predictions vs actual crisis labels
plot_prediction_timeline(trained_model, data_df, subset='developed',
                        title='Model Predictions vs True Crisis Labels')

# Analyze stock ticker against crisis periods
plot_ticker_vs_crises('SPY', 'United States', 
                      title='S&P 500 vs US Financial Crises')

# Plot multiple variables with crisis highlighting
plot_variable_vs_crises(economic_data, ['GDP', 'Unemployment'], 
                       ['United States', 'Germany'])
```

## Features

1. **Model Prediction Visualization**: Compare model predictions against actual crisis labels in timeline format
2. **Multi-Model Support**: Works with sklearn, PyTorch, and custom model architectures
3. **Stock Market Analysis**: Integrate Yahoo Finance data with crisis timelines
4. **Variable Highlighting**: Visualize economic indicators with crisis period overlays
5. **Regional Consistency**: Universal color coding system across all timeline visualizations
6. **Interactive Elements**: Support for date range limiting and period highlighting

## API Reference

### Model Prediction Timeline Functions

#### `plot_prediction_timeline(model, data_df, subset, color_by='region', title=None, date_column='Date', country_column='Country', figsize=(12, 10), bar_height=1.2, threshold=0.5)`

Plot model predictions vs actual crisis labels in a timeline format showing actual crisis periods as filled bars and predictions as X marks.

**Parameters:**
- `model`: sklearn estimator, PyTorch model, or custom model with prediction capabilities
- `data_df` (pd.DataFrame): DataFrame containing prediction data with columns [country_column, date_column, features...]
- `subset` (str): Either 'developed' or 'emerging' to load appropriate crisis data
- `color_by` (str, optional): Column name for region-based coloring. Defaults to 'region'
- `title` (str, optional): Plot title. Defaults to auto-generated title
- `date_column` (str, optional): Name of date column in data_df. Defaults to 'Date'
- `country_column` (str, optional): Name of country column. Defaults to 'Country'
- `figsize` (tuple, optional): Figure size. Defaults to (12, 10)
- `bar_height` (float, optional): Height of country bars. Defaults to 1.2
- `threshold` (float, optional): Prediction threshold for binary classification. Defaults to 0.5

**Returns:**
- None (displays plot)

**Raises:**
- `AssertionError`: If required columns are missing from DataFrames
- `Exception`: If model prediction fails for any country

**Example:**
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Prepare prediction data
data_df = pd.DataFrame({
    'Country': ['United States', 'Germany', 'United Kingdom'] * 100,
    'Date': pd.date_range('2000-01-01', periods=300, freq='M'),
    'GDP_Growth': np.random.randn(300),
    'Stock_Index': np.random.randn(300),
    'Interest_Rate': np.random.randn(300)
})

# Train model (example)
rf_model = RandomForestClassifier()
# ... training code ...

# Plot predictions vs actual crisis labels
plot_prediction_timeline(
    model=rf_model,
    data_df=data_df,
    subset='developed',
    color_by='region',
    title='Random Forest Crisis Predictions - Developed Markets',
    threshold=0.6  # Custom prediction threshold
)
```

**Model Compatibility:**
- **sklearn Models**: Uses `predict_proba()` or `predict()` methods
- **PyTorch Models**: Supports custom `predict_with_indices()` method
- **Custom Models**: Adapts to available prediction methods automatically

**Visual Elements:**
- **Crisis Labels**: Colored rectangular bars showing actual crisis periods
- **Model Predictions**: X marks indicating positive predictions
- **Regional Colors**: Consistent color coding based on geographic regions
- **Dual Y-Axes**: Country names on left, region labels on right
- **Timeline Grid**: Year markers and grid lines for temporal reference

### Financial Market Analysis Functions

#### `plot_ticker_vs_crises(ticker, country, title="", config_path='../src/config.json', figsize=(12, 4), x_limits=None, highlights=None)`

Plot ticker data from Yahoo Finance against crisis periods with optional date limits and highlighting.

**Parameters:**
- `ticker` (str): Yahoo Finance ticker symbol (e.g., 'SPY', 'AAPL', '^GSPC')
- `country` (str): Country name to get crisis data for
- `title` (str, optional): Plot title. Defaults to auto-generated
- `config_path` (str, optional): Path to configuration file. Defaults to '../src/config.json'
- `figsize` (tuple, optional): Figure size. Defaults to (12, 4)
- `x_limits` (tuple, optional): Date range as (start_date, end_date) strings. Can be one-sided
- `highlights` (list, optional): List of (start_date, end_date) tuples to highlight with cross-hatching

**Returns:**
- None (displays plot)

**Raises:**
- `FileNotFoundError`: If config file is not found
- `ConnectionError`: If Yahoo Finance data cannot be retrieved

**Example:**
```python
# Basic ticker vs crisis analysis
plot_ticker_vs_crises('SPY', 'United States', 
                      title='S&P 500 ETF vs US Financial Crises')

# Advanced analysis with date limits and highlights
plot_ticker_vs_crises(
    ticker='^GSPC',  # S&P 500 Index
    country='United States',
    title='S&P 500 Index: 2007-2009 Financial Crisis',
    x_limits=('2006-01-01', '2010-12-31'),  # Focus on crisis period
    highlights=[('2007-12-01', '2009-06-30')],  # Highlight recession
    figsize=(14, 6)
)

# One-sided date limit example
plot_ticker_vs_crises('NIKKEI', 'Japan',
                      x_limits=('2000-01-01', None),  # From 2000 onward
                      title='Nikkei Index - 21st Century Crises')
```

**Supported Ticker Formats:**
- Individual stocks: 'AAPL', 'MSFT', 'GOOGL'
- ETFs: 'SPY', 'QQQ', 'IWM' 
- Indices: '^GSPC', '^IXIC', '^RUT'
- International: 'ASML.AS', '7203.T'

**Highlight Features:**
```python
# Multiple highlight periods
highlights = [
    ('2000-03-01', '2001-12-31'),  # Dot-com crash
    ('2007-10-01', '2009-03-31'),  # Financial crisis
    ('2020-02-01', '2020-04-30')   # COVID-19 crash
]

plot_ticker_vs_crises('SPY', 'United States', highlights=highlights)
```

#### `plot_variable_vs_crises(df, columns, countries, figsize=(15,5))`

Plot specified economic variables over time with crisis period highlighting.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing time series data with 'Date', 'Country', and variable columns
- `columns` (List[str]): Names of columns/variables to plot
- `countries` (List[str]): Names of countries to include in analysis
- `figsize` (tuple, optional): Figure size. Defaults to (15, 5)

**Returns:**
- None (displays plot)

**Raises:**
- `KeyError`: If specified columns or countries are not found in DataFrame
- `ValueError`: If DataFrame structure is invalid

**Example:**
```python
# Multi-variable economic analysis
economic_indicators = pd.DataFrame({
    'Date': pd.date_range('2000-01-01', periods=300, freq='M'),
    'Country': ['United States'] * 300,
    'GDP_Growth': np.random.randn(300),
    'Unemployment_Rate': abs(np.random.randn(300)) * 5,
    'Stock_Market_Index': np.cumsum(np.random.randn(300)) + 1000,
    'Interest_Rate': abs(np.random.randn(300)) * 3
})

# Plot key economic indicators
plot_variable_vs_crises(
    df=economic_indicators,
    columns=['GDP_Growth', 'Unemployment_Rate', 'Stock_Market_Index'],
    countries=['United States'],
    figsize=(16, 8)
)

# Multi-country comparison
plot_variable_vs_crises(
    df=multi_country_data,
    columns=['GDP_Growth', 'Stock_Index'],
    countries=['United States', 'Germany', 'Japan', 'United Kingdom'],
    figsize=(20, 10)
)
```

## Advanced Features

### Model Type Detection

The timeline visualization automatically detects and adapts to different model types:

```python
# Automatic model type detection and handling
if hasattr(model, 'predict_with_indices') and hasattr(model, 'get_aligned_labels'):
    # LSTMClassifier case - handles time series alignment
    predictions, _, prediction_indices = model.predict_with_indices(country_data)
    
elif hasattr(model, 'predict') and hasattr(model, 'predict_proba') and hasattr(model, 'fnn_model'):
    # FNNClassifier case - feedforward neural network
    predictions = model.predict(country_data)
    
else:
    # Standard sklearn model case
    X_country = country_data[feature_columns]
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_country)
        positive_probs = probabilities[:, 1]
        predictions = (positive_probs >= threshold).astype(int)
    else:
        predictions = model.predict(X_country)
```

### Date Range Management

Flexible date handling for focused analysis:

```python
# Automatic date range extension for context
min_date = data_df[date_column].min() - pd.Timedelta(days=180)
max_date = data_df[date_column].max() + pd.Timedelta(days=180)

# Custom date limiting with None support
if x_limits:
    start_limit, end_limit = x_limits
    if start_limit:
        min_date = max(min_date, pd.to_datetime(start_limit))
    if end_limit:
        max_date = min(max_date, pd.to_datetime(end_limit))
```

### Crisis Period Visualization

Sophisticated crisis period rendering:

```python
# Crisis periods as filled rectangles with transparency
for _, crisis_row in country_crises.iterrows():
    crisis_year = crisis_row['Year']
    crisis_start = pd.Timestamp(f'{crisis_year}-01-01')
    crisis_end = pd.Timestamp(f'{crisis_year}-12-31')
    
    # Only plot if crisis overlaps with prediction period
    if crisis_end >= min_date and crisis_start <= max_date:
        plot_start = max(crisis_start, min_date)
        plot_end = min(crisis_end, max_date)
        
        rect = Rectangle((plot_start, i - bar_height/2), 
                        plot_end - plot_start, bar_height,
                        facecolor=color, alpha=0.3, 
                        edgecolor=color, linewidth=1)
        ax.add_patch(rect)
```

## Best Practices

### 1. Data Preparation for Timeline Analysis

```python
# Ensure proper data structure for timeline visualization
def prepare_timeline_data(raw_data):
    # Ensure datetime format
    raw_data['Date'] = pd.to_datetime(raw_data['Date'])
    
    # Sort by country and date
    timeline_data = raw_data.sort_values(['Country', 'Date'])
    
    # Verify required columns
    required_cols = ['Country', 'Date']
    assert all(col in timeline_data.columns for col in required_cols)
    
    # Remove duplicates
    timeline_data = timeline_data.drop_duplicates(['Country', 'Date'])
    
    return timeline_data

prepared_data = prepare_timeline_data(raw_prediction_data)
plot_prediction_timeline(model, prepared_data, subset='developed')
```

### 2. Model Evaluation Workflow

```python
# Comprehensive model evaluation using timeline visualization
def evaluate_model_timeline(model, test_data, subsets=['developed', 'emerging']):
    for subset in subsets:
        print(f"Evaluating {subset} markets:")
        
        # Plot predictions vs actual labels
        plot_prediction_timeline(
            model=model,
            data_df=test_data,
            subset=subset,
            title=f'Model Performance - {subset.title()} Markets',
            threshold=0.5
        )
        
        # Also create threshold sensitivity analysis
        for threshold in [0.3, 0.5, 0.7]:
            plot_prediction_timeline(
                model=model,
                data_df=test_data,
                subset=subset,
                threshold=threshold,
                title=f'{subset.title()} Markets - Threshold {threshold}'
            )

evaluate_model_timeline(crisis_model, test_dataset)
```

### 3. Market Analysis Integration

```python
# Integrate multiple analysis types for comprehensive view
def comprehensive_market_analysis(ticker, country, analysis_period):
    start_date, end_date = analysis_period
    
    # 1. Stock performance vs crises
    plot_ticker_vs_crises(
        ticker=ticker,
        country=country,
        x_limits=(start_date, end_date),
        title=f'{ticker} Performance vs {country} Financial Crises'
    )
    
    # 2. Economic indicators vs crises  
    if country_economic_data.get(country):
        plot_variable_vs_crises(
            df=country_economic_data[country],
            columns=['GDP_Growth', 'Unemployment', 'Interest_Rate'],
            countries=[country],
            figsize=(16, 6)
        )
    
    # 3. Model predictions vs actual crises
    if trained_models.get(country):
        plot_prediction_timeline(
            model=trained_models[country],
            data_df=prediction_data[country],
            subset='developed' if country in developed_countries else 'emerging',
            title=f'Crisis Prediction Model - {country}'
        )

# Example usage
comprehensive_market_analysis('SPY', 'United States', ('2005-01-01', '2012-12-31'))
```

### 4. Time Series Alignment Handling

```python
# Handle different model prediction alignment requirements
def handle_prediction_alignment(model, country_data):
    """Handle different model types and their alignment requirements."""
    
    if hasattr(model, 'predict_with_indices'):
        # LSTM models that return alignment indices
        predictions, _, indices = model.predict_with_indices(country_data)
        aligned_data = country_data.iloc[indices].copy()
        return aligned_data, predictions
    
    elif hasattr(model, 'sequence_length'):
        # Models with sequence requirements
        seq_len = model.sequence_length
        if len(country_data) < seq_len:
            return None, None
        
        aligned_data = country_data.iloc[seq_len-1:].copy()
        predictions = model.predict(country_data)
        return aligned_data, predictions
    
    else:
        # Standard models
        predictions = model.predict(country_data)
        return country_data.copy(), predictions

# Use in timeline visualization
aligned_data, preds = handle_prediction_alignment(model, country_df)
if aligned_data is not None:
    aligned_data['prediction'] = preds
    # Continue with visualization...
```

## Common Use Cases

### Crisis Prediction Model Evaluation

```python
# Evaluate multiple models on timeline visualization
models = {
    'Random Forest': rf_model,
    'LSTM': lstm_model,
    'Logistic Regression': lr_model
}

for model_name, model in models.items():
    plot_prediction_timeline(
        model=model,
        data_df=test_data,
        subset='developed',
        title=f'{model_name} - Crisis Prediction Timeline',
        figsize=(14, 8)
    )
```

### Historical Crisis Analysis

```python
# Analyze specific historical crisis events
crisis_events = {
    'Dot-com Crash': ('1999-01-01', '2002-12-31'),
    '2008 Financial Crisis': ('2006-01-01', '2010-12-31'),
    'COVID-19 Pandemic': ('2019-01-01', '2021-12-31')
}

for event_name, (start, end) in crisis_events.items():
    plot_ticker_vs_crises(
        'SPY',
        'United States', 
        title=f'S&P 500 during {event_name}',
        x_limits=(start, end),
        highlights=[(start, end)]
    )
```

### Cross-Market Comparison

```python
# Compare crisis patterns across different markets
tickers_by_country = {
    'United States': 'SPY',
    'Germany': 'EWG',
    'Japan': 'EWJ',
    'United Kingdom': 'EWU'
}

for country, ticker in tickers_by_country.items():
    plot_ticker_vs_crises(
        ticker=ticker,
        country=country,
        title=f'{country} Market Performance vs Financial Crises',
        x_limits=('2000-01-01', '2023-12-31')
    )
```

### Economic Indicator Analysis

```python
# Analyze key economic indicators around crisis periods
economic_variables = ['GDP_Growth', 'Unemployment_Rate', 'Inflation', 'Interest_Rate']
focus_countries = ['United States', 'Germany', 'Japan']

plot_variable_vs_crises(
    df=economic_panel_data,
    columns=economic_variables,
    countries=focus_countries,
    figsize=(20, 12)
)
```

## Technical Implementation Details

### Universal Color Management

Consistent color scheme across all timeline visualizations:

```python
# Global color management for consistency
_UNIVERSAL_REGION_COLORS = None

def initialize_colors():
    global _UNIVERSAL_REGION_COLORS
    if _UNIVERSAL_REGION_COLORS is None:
        _UNIVERSAL_REGION_COLORS = get_universal_region_colors()
```

### Date Formatting and Grid

Professional timeline formatting:

```python
# Set up professional date formatting
ax.xaxis.set_major_locator(mdates.YearLocator(month=7, day=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.YearLocator(month=12, day=31))

# Add grid lines at year boundaries
ax.grid(True, which='minor', axis='x', alpha=0.3, zorder=0.5)
ax.set_axisbelow(True)
```

### Dynamic Figure Sizing

Automatic figure size adjustment based on data:

```python
# Dynamic height based on number of countries
n_countries = len(countries)
figure_height = max(6, n_countries * bar_height / 4)
fig, ax = plt.subplots(figsize=(figsize[0], figure_height))
```
