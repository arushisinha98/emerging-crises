from datasets import load_dataset
import json
import pandas as pd
import seaborn as sns

import os
import dotenv
dotenv.load_dotenv()
username = os.getenv("HUGGINGFACE_USERNAME")

def load_crisis_data(subset):
    """
    Load crisis data from Hugging Face datasets.
    
    Parameters:
    -----------
    subset : 'developed' or 'emerging' to load respective crisis data.
    """
    assert subset in ['developed', 'emerging'], "Subset must be either 'developed' or 'emerging'"

    # Load ISO standard and crisis labels and merge
    iso = load_dataset(f"{username}/iso-standard-master", split='train').to_pandas()
    crises = load_dataset(f"{username}/crisis-labels-dataset", split='train').to_pandas()
    crisis_master = pd.merge(crises, iso, left_on='Country', right_on='name', how='left')

    # Load configuration file to get subset of countries present in data
    config = json.load(open('../src/config.json'))
    developed = config['DEVELOPED_MARKETS']
    emerging = config['EMERGING_MARKETS']

    # Return relevant subset of crisis dataframe
    if subset == 'developed':
        sub_df = crisis_master[crisis_master['alpha-3'].isin(developed)]
    else:
        sub_df = crisis_master[crisis_master['alpha-3'].isin(emerging)]
        rename_countries = {'Egypt, Arab Rep.': 'Egypt'} # Rename Egypt
        sub_df['Country'] = sub_df['Country'].replace(rename_countries)
    return sub_df

def get_universal_region_colors():
    """
    Get or create universal region color mapping across all datasets, avoiding red colors.
    """
    
    # Load all regions from both developed and emerging datasets
    all_regions = set()
    for subset in ['developed', 'emerging']:
        try:
            df = load_crisis_data(subset=subset)
            all_regions.update(df['region'].unique())
        except:
            continue
    
    # Sort regions for consistent ordering
    sorted_regions = sorted(list(all_regions))
    
    # Define colors excluding red
    no_red_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
        '#e377c2',  # Pink
        '#aec7e8',  # Light Blue
        '#ffbb78',  # Light Orange
        '#98df8a',  # Light Green
        '#c5b0d5',  # Light Purple
        '#c49c94',  # Light Brown
        '#f7b6d3',  # Light Pink
        '#c7c7c7',  # Light Gray
        '#dbdb8d',  # Light Olive
        '#9edae5',  # Light Cyan
        '#393b79',  # Dark Blue
        '#637939',  # Dark Green
    ]
    
    # Ensure we have enough colors for all regions
    while len(no_red_colors) < len(sorted_regions):
        # Add more blue/green/purple variations if needed
        additional_colors = sns.color_palette("Set2", n_colors=8).as_hex() + \
                           sns.color_palette("Set3", n_colors=12).as_hex()
        # Filter out red-ish colors from additional palettes
        filtered_additional = [color for color in additional_colors 
                             if not (color.startswith('#d') or color.startswith('#c') or 
                                   color.startswith('#f') or color.startswith('#e')) or
                             color in ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', 
                                      '#80b1d3', '#fdb462', '#b3de69', '#fccde5']]
        no_red_colors.extend(filtered_additional)
        break  # Prevent infinite loop
    
    # Create universal mapping
    universal_colors = {region: no_red_colors[i % len(no_red_colors)] 
                       for i, region in enumerate(sorted_regions)}
    
    return universal_colors