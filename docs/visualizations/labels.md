# Labels Visualization Documentation

The labels visualization functions in `src/visualizations/labels.py` handle
- Crisis label visualization by country and region
- Timeline-based crisis pattern analysis
- Regional comparison and highlighting of concurrent crisis periods

The utilities are designed to work with crisis label datasets and support region-based coloring for clear visual analysis of financial crisis patterns across different economic regions.

## Usage Instructions

```python
from src.visualizations.labels import plot_crises_labels

# Plot crisis labels for developed markets
plot_crises_labels(subset='developed', 
                   color_by='region',
                   title='Financial Crisis Labels - Developed Markets')

# Highlight concurrent crisis periods and specific countries
plot_crises_labels(subset='emerging',
                   geq=3,
                   special_countries=['Brazil', 'Mexico'],
                   title='Emerging Market Crisis Patterns')
```

## Features

1. **Regional Color Coding**: Color assignment based on geographic regions using universal color mapping
2. **Concurrent Crisis Highlighting**: Visual highlighting of years with multiple simultaneous crises
3. **Country-Specific Emphasis**: Optional highlighting of specific countries of interest
4. **Timeline Visualization**: Clear timeline display of crisis periods across countries
5. **Consistent Styling**: Universal color scheme for consistency across all visualizations

## API Reference

### Crisis Label Visualization Functions

#### `plot_crises_labels(subset, color_by='region', bar_height=1, geq=3, title=None, special_countries=[])`

Plot crisis labels by country with regions colored in a timeline format.

**Parameters:**
- `subset` (str): Either 'developed' or 'emerging' to load the appropriate crisis data
- `color_by` (str, optional): Column name to use for region-based coloring. Defaults to 'region'
- `bar_height` (float, optional): Height of the country bars. Defaults to 1
- `geq` (int, optional): Minimum number of concurrent crises in a year to highlight with background shading. Defaults to 3
- `title` (str, optional): Plot title. If None, uses default title
- `special_countries` (list, optional): List of countries to highlight with grey background shading. Defaults to []

**Returns:**
- None (displays plot)

**Raises:**
- `AssertionError`: If `color_by` column is not present in the crisis dataset
- `ValueError`: If `subset` is not 'developed' or 'emerging'

**Example:**
```python
# Basic crisis label visualization
plot_crises_labels(subset='developed', 
                   title='Financial Crisis Labels - Developed Markets')

# Advanced visualization with concurrent crisis highlighting
plot_crises_labels(subset='emerging',
                   color_by='region',
                   bar_height=1.2,
                   geq=4,  # Highlight years with 4+ concurrent crises
                   special_countries=['Brazil', 'Mexico', 'Thailand'],
                   title='Emerging Market Crisis Patterns with OECD Countries Highlighted')

# Regional comparison with custom parameters
plot_crises_labels(subset='developed',
                   color_by='region',
                   bar_height=0.8,
                   geq=2,
                   title='Developed Market Regional Crisis Analysis')
```

**Visual Elements:**
- **Crisis Bars**: Colored horizontal bars representing crisis periods for each country
- **Regional Colors**: Consistent color coding based on geographic regions
- **Background Shading**: Grey vertical bands for years with ≥ specified concurrent crises
- **Country Highlighting**: Grey horizontal bands for special countries of interest
- **Dual Y-Axes**: Country names on left, region labels on right for clear identification

**Data Requirements:**
The function expects crisis data to contain:
- `Country`: Country names
- `Year`: Crisis years
- `region` (or specified `color_by` column): Regional classifications

**Regional Color Scheme:**
The function uses a universal color mapping system that ensures consistency across all visualizations:

```python
# Example of universal color mapping
_UNIVERSAL_REGION_COLORS = {
    'Europe': '#1f77b4',           # Blue
    'North America': '#ff7f0e',    # Orange  
    'Asia': '#2ca02c',             # Green
    'Latin America': '#bcbd22',    # Olive
    'Middle East': '#17becf',      # Cyan
    # ... additional regions
}
```

## Visualization Features

### Timeline Display

The plot displays crisis data in a horizontal timeline format:

```python
# Crisis periods are shown as horizontal bars
# Each country gets one row
# X-axis shows years
# Y-axis shows countries (grouped by region)
```

### Concurrent Crisis Highlighting

Years with multiple simultaneous crises are automatically highlighted:

```python
# Count crises per year
crises_count_per_year = df.groupby('Year')['Country'].nunique()
years_geq = crises_count_per_year[crises_count_per_year >= geq].index.tolist()

# Add background shading for these years
for year in years_geq:
    ax.axvspan(year, year + 1, alpha=0.2, color='grey', zorder=0)
```

### Legend Management

The function creates comprehensive legends that include:

1. **Regional Legend**: Shows all regions with their assigned colors
2. **Special Highlighting Legend**: Documents any special country highlighting
3. **Concurrent Crisis Legend**: Explains background shading for concurrent crisis periods

## Best Practices

### 1. Data Preparation

```python
# Ensure crisis data is properly structured
crisis_data = load_crisis_data(subset='developed')

# Verify required columns exist
required_columns = ['Country', 'Year', 'region']
for col in required_columns:
    assert col in crisis_data.columns, f"Missing required column: {col}"

# Check data quality
assert crisis_data['Year'].min() >= 1800, "Unrealistic crisis years detected"
assert crisis_data['Country'].notna().all(), "Missing country names found"
```

### 2. Regional Analysis Workflow

```python
# Compare crisis patterns across different market types
market_types = ['developed', 'emerging']

for market_type in market_types:
    plot_crises_labels(
        subset=market_type,
        color_by='region',
        geq=3,
        title=f'Crisis Patterns - {market_type.title()} Markets'
    )
```

### 3. Highlighting Countries of Interest

```python
# Highlight specific countries for focused analysis
oecd_countries = ['United States', 'Germany', 'Japan', 'United Kingdom']
brics_countries = ['Brazil', 'Russia', 'India', 'China', 'South Africa']

# OECD analysis
plot_crises_labels(
    subset='developed',
    special_countries=oecd_countries,
    title='Crisis Patterns with OECD Core Countries Highlighted'
)

# BRICS analysis  
plot_crises_labels(
    subset='emerging',
    special_countries=brics_countries,
    title='Emerging Market Crises with BRICS Countries Highlighted'
)
```

### 4. Concurrent Crisis Analysis

```python
# Analyze different thresholds for concurrent crises
thresholds = [2, 3, 4, 5]

for threshold in thresholds:
    plot_crises_labels(
        subset='developed',
        geq=threshold,
        title=f'Concurrent Crisis Analysis - {threshold}+ Countries'
    )
```

## Technical Implementation Details

### Color Management System

The module uses a global color management system for consistency:

```python
_UNIVERSAL_REGION_COLORS = None

def plot_crises_labels(...):
    global _UNIVERSAL_REGION_COLORS
    
    # Initialize universal colors if not already done
    if _UNIVERSAL_REGION_COLORS is None:
        _UNIVERSAL_REGION_COLORS = get_universal_region_colors()
```

### Bar Plotting Algorithm

Crisis periods are visualized using matplotlib's `broken_barh` function:

```python
# Convert crisis years to (start, duration) tuples
crisis_years = sorted(country_data['Year'].unique())
xranges = [(year, 1) for year in crisis_years]

# Plot the bars
ax.broken_barh(xranges, (i-bar_height/2, bar_height), 
               facecolors=country_colors[country], 
               alpha=1,
               linewidth=0.5,
               zorder=1)
```

### Dual Y-Axis Setup

The visualization uses dual y-axes for country names and region labels:

```python
# Set country labels on y-axis (left side)
ax.set_yticks(range(len(countries)))
ax.set_yticklabels(countries)

# Create right y-axis for region labels
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(len(countries)))
ax2.set_yticklabels(region_labels)
```

## Common Use Cases

### Regional Crisis Pattern Analysis

```python
# Analyze crisis patterns by geographic region
for subset in ['developed', 'emerging']:
    plot_crises_labels(
        subset=subset,
        color_by='region',
        bar_height=1,
        geq=3,
        title=f'{subset.title()} Markets - Regional Crisis Patterns'
    )
```

### Historical Crisis Event Study

```python
# Study specific historical crisis periods
# Highlight major crisis years (e.g., 2008 Financial Crisis)
plot_crises_labels(
    subset='developed',
    geq=3,  # Show years with 3+ concurrent crises
    title='2008 Financial Crisis - Developed Market Impact'
)

plot_crises_labels(
    subset='emerging', 
    geq=2,  # Lower threshold for emerging markets
    title='Global Crisis Contagion - Emerging Market Impact'
)
```

### Country-Specific Analysis

```python
# Focus on specific countries or economic groups
g7_countries = ['United States', 'Germany', 'Japan', 'United Kingdom', 
                'France', 'Italy', 'Canada']

plot_crises_labels(
    subset='developed',
    special_countries=g7_countries,
    geq=4,
    title='G7 Countries Crisis Timeline Analysis'
)
```

### Comparative Regional Studies

```python
# Compare crisis frequency across regions
def analyze_region_crisis_frequency():
    for subset in ['developed', 'emerging']:
        crisis_data = load_crisis_data(subset=subset)
        
        # Count crises by region
        region_crisis_counts = crisis_data.groupby('region')['Country'].nunique()
        print(f"\n{subset.title()} Markets - Crises by Region:")
        print(region_crisis_counts.sort_values(ascending=False))
        
        # Visualize
        plot_crises_labels(
            subset=subset,
            color_by='region',
            title=f'{subset.title()} Markets - Regional Distribution'
        )

analyze_region_crisis_frequency()
```

### Temporal Analysis

```python
# Analyze crisis patterns over different time periods
def analyze_temporal_patterns():
    # Define time periods of interest
    periods = {
        'Pre-2000': (1980, 1999),
        '2000s': (2000, 2009), 
        '2010s': (2010, 2019),
        '2020s': (2020, 2030)
    }
    
    for period_name, (start_year, end_year) in periods.items():
        plot_crises_labels(
            subset='developed',
            geq=2,
            title=f'Crisis Patterns - {period_name} ({start_year}-{end_year})'
        )

analyze_temporal_patterns()
```

## Integration with Other Visualization Modules

### Consistent Color Scheme

The labels module integrates with other visualization modules through shared color management:

```python
# Color consistency across all visualizations
from .utilities import get_universal_region_colors

# This ensures timeline.py, labels.py use same colors
universal_colors = get_universal_region_colors()
```

### Data Pipeline Integration

```python
# Typical workflow: labels → timeline → model evaluation
# 1. Visualize crisis labels
plot_crises_labels(subset='developed', title='Crisis Labels')

# 2. Compare with model predictions  
from .timeline import plot_prediction_timeline
plot_prediction_timeline(model, data_df, subset='developed')

# 3. Evaluate model performance
from .auc import plot_roc_curves
plot_roc_curves(results_df, title='Model Performance')
```
