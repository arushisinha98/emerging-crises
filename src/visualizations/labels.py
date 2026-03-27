import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from .utilities import load_crisis_data, get_universal_region_colors

_UNIVERSAL_REGION_COLORS = None

def plot_crises_labels(subset, color_by='region',
                       bar_height=1, geq=3,
                       title=None,
                       special_countries=[]):
    """
    Plot crisis labels by country with regions colored.
    
    Parameters:
    -----------
    subset : str
        Either 'developed' or 'emerging' to load the appropriate crisis data
    color_by : str
        Column name to use for region-based coloring (default: 'region')
    bar_height : float
        Height of the country bars (default: 1)
    geq : int
        Minimum number of concurrent crises in a year to highlight (default: 3)
    title : str
        Plot title (default: None)
    special_countries : list
        List of countries to highlight with grey shading (default: [])
    """
    global _UNIVERSAL_REGION_COLORS
    # Initialize universal colors if not already done
    if _UNIVERSAL_REGION_COLORS is None:
        _UNIVERSAL_REGION_COLORS = get_universal_region_colors()
    
    df = load_crisis_data(subset=subset)
    assert color_by in df.columns, f"Color by '{color_by}' must be a column in the DataFrame"
    
    # Get unique regions and countries
    regions = df[color_by].unique()
    countries_df = df.groupby([color_by, 'Country']).size().reset_index()
    countries_df = countries_df.sort_values([color_by, 'Country'])
    countries = countries_df['Country'].unique()
    
    # Use universal color mapping
    region_colors = {region: _UNIVERSAL_REGION_COLORS.get(region, 'gray') 
                    for region in regions}

    # Assign colors to countries based on their region
    country_colors = {}
    for country in countries:
        country_region = df[df['Country'] == country][color_by].iloc[0]
        country_colors[country] = region_colors[country_region]
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(12, len(countries) * bar_height/4))
    legend_elements = []
    
    # Count countries per year to identify years with >= X crises
    if geq > 1:
        crises_count_per_year = df.groupby('Year')['Country'].nunique()
        years_geq = crises_count_per_year[crises_count_per_year >= geq].index.tolist()

        # Add grey background shading for years with >= X countries in crisis
        for year in years_geq:
            ax.axvspan(year, year + 1, alpha=0.2, color='grey', zorder=0)

    # Plot each country's crises using broken_barh
    for i, country in enumerate(countries):
        country_data = df[df['Country'] == country]
        
        # Convert crisis years to (start, duration) tuples
        crisis_years = sorted(country_data['Year'].unique())
        xranges = [(year, 1) for year in crisis_years]

        if country in special_countries:
            ax.axhspan(i - bar_height/2, i + bar_height/2, color='grey', alpha=0.3, zorder=0)

        # Plot the bars
        ax.broken_barh(xranges, (i-bar_height/2, bar_height), 
                       facecolors=country_colors[country], 
                       alpha=1,
                       linewidth=0.5,
                       zorder=1)
        
    # Add legend elements for regions
    all_regions_sorted = sorted(_UNIVERSAL_REGION_COLORS.keys())
    for region in all_regions_sorted:
        if region in regions:
            label = f"Crisis Labels in {region}"
            alpha = 1.0
            legend_elements.append(Patch(facecolor=_UNIVERSAL_REGION_COLORS[region], 
                                        alpha=alpha,
                                        label=label))
    
    # Add shading legend if there are special countries / years
    if special_countries:
        legend_elements.append(Patch(facecolor='grey', alpha=0.3, 
                                     label="OECD countries"))
    if geq > 1:
        if years_geq:
            legend_elements.append(Patch(facecolor='grey', alpha=0.2, 
                                         label=rf"Years with $\geq${geq} concurrent crisis labels"))
    
    # Set x-axis limits to cover the full range of years
    ax.set_xlim(df['Year'].min() - 1, df['Year'].max() + 1)
    ax.set_ylim(-0.5, len(countries) - 0.5)
    ax.set_xlabel('Year', fontsize=12)
    
    # Set country labels on y-axis (left side)
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries)
    
    # Create right y-axis for region labels
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    
    # Find first occurrence of each region and create labels
    region_labels = []
    region_positions = []
    seen_regions = set()
    
    for i, country in enumerate(countries):
        country_region = df[df['Country'] == country][color_by].iloc[0]
        if country_region not in seen_regions:
            region_labels.append(country_region)
            region_positions.append(i)
            seen_regions.add(country_region)
        else:
            region_labels.append('')
            region_positions.append(i)
    
    # Set the right y-axis ticks and labels
    ax2.set_yticks(range(len(countries)))
    ax2.set_yticklabels([region_labels[i] if region_labels[i] else '' for i in range(len(countries))])
    ax2.invert_yaxis() 
    
    ax.grid(True, axis='x', alpha=0.3, zorder=0.5)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    
    # Add legend below the plot
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 100*(-0.05)/len(countries)), 
                 ncol=min(len(legend_elements), 4), frameon=False)
    
    plt.title(title if title else "Crisis Labels by Country", fontsize=14)
    plt.tight_layout()
    plt.show()