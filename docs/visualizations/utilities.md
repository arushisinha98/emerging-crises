# Visualization Utilities Documentation

The visualization utilities in `src/visualizations/utilities.py` handle
- Crisis data loading and preprocessing from HuggingFace datasets
- Universal color scheme management for consistent visualizations
- ISO standard integration for geographic region mapping
- Configuration-based data filtering for market subset analysis

The utilities provide the foundational data access and styling functions that support all other visualization modules in the package.

## Usage Instructions

```python
from src.visualizations.utilities import (
    load_crisis_data,
    get_universal_region_colors
)

# Load crisis data for developed markets
developed_crisis_data = load_crisis_data(subset='developed')

# Get consistent color mapping for regions
region_colors = get_universal_region_colors()
```

## Features

1. **Unified Data Loading**: Consistent interface for accessing crisis datasets across market types
2. **Geographic Integration**: Merging of ISO standards with crisis data for regional analysis
3. **Universal Color Management**: Centralized color scheme ensuring consistency across all visualizations
4. **Market Filtering**: Configuration-driven filtering for developed vs emerging market analysis
5. **Data Quality Assurance**: Validation and data cleaning for reliable visualizations

## API Reference

### Data Loading Functions

#### `load_crisis_data(subset)`

Load crisis data from HuggingFace datasets with integrated ISO standard geographic information.

**Parameters:**
- `subset` (str): Market subset to load. Must be either 'developed' or 'emerging'

**Returns:**
- `pd.DataFrame`: Crisis data with merged geographic and regional information containing columns:
  - `Country`: Country names
  - `Year`: Crisis years  
  - `name`: Standardized country name from ISO
  - `alpha-3`: ISO 3-letter country code
  - `region`: Geographic region classification
  - Additional ISO standard fields (numeric codes, regions, subregions)

**Raises:**
- `AssertionError`: If subset is not 'developed' or 'emerging'
- `ConnectionError`: If HuggingFace datasets cannot be accessed
- `KeyError`: If required environment variables are missing

**Example:**
```python
# Load developed market crisis data
developed_data = load_crisis_data(subset='developed')
print(developed_data.head())
#           Country  Year        name alpha-3       region
# 0   United States  2008  United States     USA  North America
# 1         Germany  2008     Germany     DEU       Europe
# 2           Japan  1997      Japan     JPN         Asia

# Load emerging market crisis data
emerging_data = load_crisis_data(subset='emerging')
print(f"Emerging markets: {emerging_data['Country'].unique()}")

# Analyze regional distribution
regional_breakdown = developed_data['region'].value_counts()
print(regional_breakdown)
```

**Data Integration Process:**
1. **Crisis Labels**: Loads from `{username}/crisis-labels-dataset`
2. **ISO Standards**: Loads from `{username}/iso-standard-master`
3. **Configuration**: Filters based on `config.json` market definitions
4. **Merging**: Combines datasets using country name matching
5. **Filtering**: Returns only countries present in specified market subset

**Market Subset Configuration:**
The function relies on configuration file definitions:
```json
{
    "DEVELOPED_MARKETS": ["USA", "GBR", "DEU", "JPN", ...],
    "EMERGING_MARKETS": ["BRA", "CHN", "IND", "RUS", ...]
}
```

**Data Quality Features:**
- **Country Name Standardization**: Handles name variations (e.g., 'Egypt, Arab Rep.' → 'Egypt')
- **ISO Code Validation**: Ensures proper ISO 3166 country code mapping
- **Missing Data Handling**: Provides warnings for countries without ISO matches
- **Regional Classification**: Consistent geographic region assignment

### Color Management Functions

#### `get_universal_region_colors()`

Get or create universal region color mapping across all datasets, ensuring consistency across visualizations while avoiding problematic color choices.

**Parameters:**
- None

**Returns:**
- `dict`: Dictionary mapping region names to hex color codes
  ```python
  {
      'Europe': '#1f77b4',           # Blue
      'North America': '#ff7f0e',    # Orange
      'Asia': '#2ca02c',             # Green
      'Latin America': '#bcbd22',    # Olive
      'Middle East': '#17becf',      # Cyan
      # ... additional regions
  }
  ```

**Raises:**
- `ConnectionError`: If unable to access datasets for region discovery
- `ImportError`: If required visualization libraries are not available

**Example:**
```python
# Get universal color mapping
region_colors = get_universal_region_colors()

# Use in visualizations
import matplotlib.pyplot as plt

for region, color in region_colors.items():
    # Plot data for this region using consistent color
    region_data = crisis_data[crisis_data['region'] == region]
    plt.scatter(region_data['Year'], region_data['Crisis_Intensity'], 
               color=color, label=region)

plt.legend()
plt.show()

# Colors are consistent across all visualization modules
print(f"Europe color: {region_colors['Europe']}")  # Always #1f77b4
```

**Color Selection Principles:**
1. **Accessibility**: Colors chosen for colorblind accessibility
2. **Distinction**: High contrast between adjacent regions
3. **Consistency**: Same region always gets same color across all plots
4. **Professional**: Appropriate colors for academic/financial contexts
5. **Red Avoidance**: Excludes red colors that are too jarring

**Color Palette Strategy:**
```python
# Primary color palette (excluding red tones)
no_red_colors = [
    '#1f77b4',  # Blue - Professional, trustworthy
    '#ff7f0e',  # Orange - Energetic but not alarming  
    '#2ca02c',  # Green - Positive, stable
    '#bcbd22',  # Olive - Neutral, sophisticated
    '#17becf',  # Cyan - Cool, analytical
    '#e377c2',  # Pink - Distinctive without being red
    # ... additional colors
]
```

**Dynamic Region Discovery:**
The function automatically discovers all regions across both developed and emerging datasets:

```python
# Automatic region discovery
all_regions = set()
for subset in ['developed', 'emerging']:
    try:
        df = load_crisis_data(subset=subset)
        all_regions.update(df['region'].unique())
    except:
        continue

# Consistent alphabetical ordering
sorted_regions = sorted(list(all_regions))
```

## Technical Implementation Details

### Data Loading Architecture

The utilities implement a robust data loading system:

```python
def load_crisis_data(subset):
    # Environment validation
    username = os.getenv("HUGGINGFACE_USERNAME")
    if not username:
        raise ValueError("HUGGINGFACE_USERNAME environment variable required")
    
    # Load core datasets
    iso = load_dataset(f"{username}/iso-standard-master", split='train').to_pandas()
    crises = load_dataset(f"{username}/crisis-labels-dataset", split='train').to_pandas()
    
    # Merge with geographic data
    crisis_master = pd.merge(crises, iso, left_on='Country', right_on='name', how='left')
    
    # Apply configuration filtering
    config = json.load(open('../src/config.json'))
    market_countries = config[f'{subset.upper()}_MARKETS']
    
    # Filter and clean data
    filtered_data = crisis_master[crisis_master['alpha-3'].isin(market_countries)]
    
    return filtered_data
```

### Universal Color Management System

The color system ensures consistency through global state management:

```python
# Global color cache for consistency
_region_color_cache = None

def get_universal_region_colors():
    global _region_color_cache
    
    if _region_color_cache is None:
        # Discover all regions
        all_regions = discover_all_regions()
        
        # Generate consistent color mapping
        _region_color_cache = generate_color_mapping(all_regions)
    
    return _region_color_cache
```

### Configuration Integration

Seamless integration with project configuration:

```python
# Configuration file structure expected
{
    "DEVELOPED_MARKETS": ["USA", "GBR", "DEU", "JPN", "FRA", "ITA", "CAN"],
    "EMERGING_MARKETS": ["BRA", "CHN", "IND", "RUS", "ZAF", "MEX", "TUR"],
    "DATA_PATHS": {...},
    "MODEL_PARAMETERS": {...}
}
```

## Best Practices

### 1. Environment Setup

```python
# Ensure proper environment configuration
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify required variables
required_vars = ['HUGGINGFACE_USERNAME', 'HUGGINGFACE_TOKEN']
for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"Required environment variable {var} is not set")

# Now safe to use utilities
crisis_data = load_crisis_data('developed')
```

### 2. Data Loading Workflow

```python
# Standard data loading and validation workflow
def load_and_validate_crisis_data(subset):
    """Load crisis data with comprehensive validation."""
    
    # Load data
    data = load_crisis_data(subset=subset)
    
    # Validate structure
    required_columns = ['Country', 'Year', 'region', 'alpha-3']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate data quality
    assert data['Year'].min() >= 1800, "Unrealistic crisis years detected"
    assert data['Country'].notna().all(), "Missing country names found"
    assert data['alpha-3'].str.len().eq(3).all(), "Invalid ISO codes found"
    
    print(f"Loaded {len(data)} crisis records for {subset} markets")
    print(f"Countries: {data['Country'].nunique()}")
    print(f"Regions: {data['region'].nunique()}")
    print(f"Year range: {data['Year'].min()}-{data['Year'].max()}")
    
    return data

# Use in analysis
developed_data = load_and_validate_crisis_data('developed')
emerging_data = load_and_validate_crisis_data('emerging')
```

### 3. Color Consistency Management

```python
# Ensure color consistency across multiple visualizations
def create_consistent_visualizations():
    """Create multiple visualizations with consistent colors."""
    
    # Get universal colors once
    universal_colors = get_universal_region_colors()
    
    # Use in multiple plot types
    datasets = ['developed', 'emerging']
    
    for dataset in datasets:
        crisis_data = load_crisis_data(subset=dataset)
        
        # Timeline visualization
        plot_crises_labels(subset=dataset, title=f'{dataset.title()} Markets')
        
        # Regional analysis
        for region in crisis_data['region'].unique():
            region_data = crisis_data[crisis_data['region'] == region]
            color = universal_colors[region]
            
            plt.scatter(region_data['Year'], range(len(region_data)), 
                       color=color, label=region, alpha=0.7)
        
        plt.legend()
        plt.title(f'Regional Crisis Distribution - {dataset.title()}')
        plt.show()

create_consistent_visualizations()
```

## Integration with Other Modules

### Visualization Module Integration

The utilities serve as the foundation for all visualization modules:

```python
# labels.py integration
from .utilities import load_crisis_data, get_universal_region_colors

def plot_crises_labels(subset, **kwargs):
    # Use utilities for data loading
    df = load_crisis_data(subset=subset)
    colors = get_universal_region_colors()
    # ... visualization code

# timeline.py integration  
from .utilities import load_crisis_data, get_universal_region_colors

def plot_prediction_timeline(model, data_df, subset, **kwargs):
    # Use utilities for crisis data
    crisis_df = load_crisis_data(subset=subset)
    colors = get_universal_region_colors()
    # ... visualization code
```

### Configuration System Integration

```python
# Integration with project configuration
import json

def get_market_configuration():
    """Get market configuration from project config."""
    config_path = '../src/config.json'
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return {
            'developed': config['DEVELOPED_MARKETS'],
            'emerging': config['EMERGING_MARKETS']
        }
    
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        raise
    
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in configuration file: {e}")
        raise

# Use in utilities
market_config = get_market_configuration()
```

## Common Use Cases

### Regional Analysis Setup

```python
# Set up comprehensive regional analysis
def setup_regional_analysis():
    """Initialize data and colors for regional analysis."""
    
    # Load data for both market types
    developed_data = load_crisis_data('developed')
    emerging_data = load_crisis_data('emerging')
    
    # Get consistent colors
    region_colors = get_universal_region_colors()
    
    # Combine for comprehensive analysis
    all_data = pd.concat([developed_data, emerging_data], ignore_index=True)
    
    # Regional summaries
    regional_stats = all_data.groupby('region').agg({
        'Country': 'nunique',
        'Year': ['min', 'max', 'count']
    }).round(2)
    
    print("Regional Crisis Statistics:")
    print(regional_stats)
    
    return all_data, region_colors

# Use for analysis
crisis_data, colors = setup_regional_analysis()
```

### Data Quality Assessment

```python
# Comprehensive data quality assessment
def assess_data_quality():
    """Perform comprehensive data quality assessment."""
    
    quality_report = {}
    
    for subset in ['developed', 'emerging']:
        data = load_crisis_data(subset)
        
        quality_metrics = {
            'total_records': len(data),
            'unique_countries': data['Country'].nunique(),
            'unique_years': data['Year'].nunique(),
            'year_range': (data['Year'].min(), data['Year'].max()),
            'missing_regions': data['region'].isna().sum(),
            'missing_iso_codes': data['alpha-3'].isna().sum(),
            'duplicate_records': data.duplicated(['Country', 'Year']).sum()
        }
        
        quality_report[subset] = quality_metrics
    
    # Print report
    for subset, metrics in quality_report.items():
        print(f"\n{subset.upper()} MARKETS DATA QUALITY:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    return quality_report

# Run assessment
quality_results = assess_data_quality()
```

### Cross-Dataset Consistency Checks

```python
# Verify consistency across datasets and configurations
def verify_data_consistency():
    """Verify consistency between datasets and configuration."""
    
    # Load configuration
    config = json.load(open('../src/config.json'))
    
    # Check developed markets
    developed_data = load_crisis_data('developed')
    developed_countries_in_data = set(developed_data['alpha-3'].unique())
    developed_countries_in_config = set(config['DEVELOPED_MARKETS'])
    
    # Check emerging markets
    emerging_data = load_crisis_data('emerging')
    emerging_countries_in_data = set(emerging_data['alpha-3'].unique())
    emerging_countries_in_config = set(config['EMERGING_MARKETS'])
    
    # Report discrepancies
    print("CONSISTENCY CHECK RESULTS:")
    
    developed_missing = developed_countries_in_config - developed_countries_in_data
    if developed_missing:
        print(f"Developed markets in config but missing from data: {developed_missing}")
    
    emerging_missing = emerging_countries_in_config - emerging_countries_in_data
    if emerging_missing:
        print(f"Emerging markets in config but missing from data: {emerging_missing}")
    
    # Check for overlaps
    overlap = developed_countries_in_config & emerging_countries_in_config
    if overlap:
        print(f"Countries in both developed and emerging configs: {overlap}")
    
    print("Consistency check completed.")

verify_data_consistency()
```
