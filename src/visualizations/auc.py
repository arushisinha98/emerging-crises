import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from sklearn.metrics import roc_curve, auc

def plot_roc_curves(
        results_df: pd.DataFrame,
        linestyle_map: dict = None,
        title: str = 'AUC-ROC Curves',
        figsize: tuple = (8, 6),
        label: str = None):
    """
    Plot multiple ROC curves on the same plot for model comparison.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame of model predictions where index is y_true and each column contains 
        y_pred_proba for different models
    linestyle_map : dict
        Map of column names to matplotlib linestyle strings
        (e.g., {'model1': '-b', 'model2': '--r', 'model3': ':g'})
    title : str
        Plot title
    figsize : tuple
        Figure size as (width, height)
    label : str
        Text label to display in the top left corner of the plot
    """
    if linestyle_map is None:
        # Default colors and linestyles if none provided
        colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
        linestyles = ['-', '--', '-.', ':']
        linestyle_map = {}
        for i, col in enumerate(results_df.columns):
            color = colors[i % len(colors)]
            style = linestyles[i % len(linestyles)]
            linestyle_map[col] = f'{style}{color}'
    
    plt.figure(figsize=figsize)
    
    y_true = results_df.index.values
    
    # Plot ROC curve for each model
    for model_name in results_df.columns:
        y_pred_proba = results_df[model_name].values
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Extract color and style from linestyle string
        linestyle_str = linestyle_map.get(model_name, '-b')
        
        plt.plot(fpr, tpr, linestyle_str, lw=2, 
                 label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot random guess line
    plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, label='Random guess')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Add label text in top left corner if provided
    if label:
        plt.text(0.05, 0.95, label, transform=plt.gca().transAxes, 
                fontsize=16, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()


def plot_auc_roc_scatter(data: dict, figsize=(10, 6)):
    """
    Plot dimensionality reduction AUC-ROC results with separate subplots for each method.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing AUC-ROC results for different methods and experiments.
        Values can be None to skip connecting lines.
    figsize : tuple
        Figure size (width, height)
    """
    # Extract groups and clean method names
    groups = {}
    cleaned_methods = []
    
    for method in data.keys():
        if '[' in method and ']' in method:
            # Extract group text and clean method name
            start_bracket = method.find('[')
            end_bracket = method.find(']')
            group_text = method[start_bracket+1:end_bracket]
            clean_method = method[:start_bracket].strip()
            
            if group_text not in groups:
                groups[group_text] = []
            groups[group_text].append((clean_method, method))
        else:
            # No group
            if 'ungrouped' not in groups:
                groups['ungrouped'] = []
            groups['ungrouped'].append((method, method))
        
        cleaned_methods.append(method)
    
    # Create ordered list of methods maintaining original order
    ordered_methods = []
    group_positions = {}
    current_pos = 0
    
    for group_name, methods_in_group in groups.items():
        group_positions[group_name] = (current_pos, current_pos + len(methods_in_group) - 1)
        for clean_name, original_name in methods_in_group:
            ordered_methods.append((clean_name, original_name))
        current_pos += len(methods_in_group)
    
    # Set up the subplots
    fig, axes = plt.subplots(len(ordered_methods), 1, figsize=figsize, sharex=True)
    if len(ordered_methods) == 1:
        axes = [axes]  # Ensure axes is always a list
    fig.suptitle('AUC-ROC by Dimensionality Reduction Method and Train-Test Configuration', fontsize=14)
    
    # Define markers for training data (unfilled)
    developed_only_marker = 'o'
    emerging_only_marker = '^'
    mixed_marker = 'D'
    marker_color = '#1f77b4'
    
    for i, (clean_method, original_method) in enumerate(ordered_methods):
        if original_method not in data:
            continue
            
        results = data[original_method]
        axes[i].set_title(f'{clean_method}', fontsize=12)
        
        # Add subplot label (a), (b), (c), etc.
        label = chr(ord('a') + i)
        axes[i].text(0.01, 0.95, f'({label})', transform=axes[i].transAxes, 
                    fontsize=12, verticalalignment='top')
        
        # Data containers for connecting lines (only store non-None values)
        dev_test_points = []
        emg_test_points = []
        
        # Y positions for different test markets
        y_pos_developed = 0.6
        y_pos_emerging = 0.4
        
        # Process each experiment
        for experiment, auc_roc in results.items():
            # Skip None values for plotting
            if auc_roc is None:
                continue
                
            # Parse experiment string
            parts = experiment.split(', ')
            train_data = parts[0]
            test_data = parts[1]
            
            # Determine marker and position based on training data
            if '+' in train_data:  # Mixed training (Developed + Emerging)
                marker = mixed_marker
                train_type = 'Mixed'
                face_color = marker_color
                edge_color = marker_color
            else:  # Single market training
                if 'Developed' in train_data:
                    train_type = 'Developed'
                    marker = developed_only_marker
                else:  # Emerging only
                    train_type = 'Emerging'
                    marker = emerging_only_marker
                face_color = 'none'
                edge_color = marker_color
            
            # Plot on appropriate y-position based on test data
            if test_data == 'Developed':
                y_pos = y_pos_developed
                axes[i].scatter(auc_roc, y_pos, marker=marker, facecolors=face_color, 
                              edgecolors=edge_color, s=80, linewidth=2, zorder=3)

                if train_type == 'Mixed':
                    xytext = (0, 15)
                else:
                    xytext = (0, -15)
                ha = 'center'
                
                axes[i].annotate(f'{auc_roc:.2f}', (auc_roc, y_pos), 
                               textcoords="offset points", xytext=xytext, 
                               ha=ha, va='center', fontsize=9, weight='bold')
                dev_test_points.append((train_type, auc_roc, y_pos))
                
            else:  # Emerging
                y_pos = y_pos_emerging
                axes[i].scatter(auc_roc, y_pos, marker=marker, facecolors=face_color, 
                              edgecolors=edge_color, s=80, linewidth=2, zorder=3)

                if train_type == 'Mixed':
                    xytext = (0, 15)
                else:
                    xytext = (0, -15)
                ha = 'center'
                
                axes[i].annotate(f'{auc_roc:.2f}', (auc_roc, y_pos), 
                               textcoords="offset points", xytext=xytext, 
                               ha=ha, va='center', fontsize=9, weight='bold')
                emg_test_points.append((train_type, auc_roc, y_pos))
        
        # Draw connecting lines for developed test results (only if we have exactly 2 non-None points)
        if len(dev_test_points) == 2:
            dev_test_points.sort(key=lambda x: x[1])  # Sort by AUC score
            x1, y1 = dev_test_points[0][1], dev_test_points[0][2]
            x2, y2 = dev_test_points[1][1], dev_test_points[1][2]
            axes[i].plot([x1, x2], [y1, y2], 'k-', alpha=0.5, linewidth=2, zorder=2)
        
        # Draw connecting lines for emerging test results (only if we have exactly 2 non-None points)
        if len(emg_test_points) == 2:
            emg_test_points.sort(key=lambda x: x[1])  # Sort by AUC score
            x1, y1 = emg_test_points[0][1], emg_test_points[0][2]
            x2, y2 = emg_test_points[1][1], emg_test_points[1][2]
            axes[i].plot([x1, x2], [y1, y2], 'k-', alpha=0.5, linewidth=2, zorder=2)
        
        # Customize subplots
        axes[i].set_xlim(0.4, 1.0)
        axes[i].set_ylim(0.25, 0.75)
        axes[i].set_yticks([0.4, 0.6])
        axes[i].set_yticklabels(['Emerging', 'Developed'], fontsize=12)
        axes[i].grid(True, alpha=0.3, axis='x')
        axes[i].set_axisbelow(True)
    
    # Set xlabel only on the bottom subplot
    axes[-1].set_xlabel('AUC-ROC', fontsize=12)
    
    # Create legend
    legend_elements = [
        plt.scatter([], [], marker=developed_only_marker, facecolors='none', 
                   edgecolors=marker_color, s=80, linewidth=2,
                   label='Trained on Developed Markets Only'),
        plt.scatter([], [], marker=emerging_only_marker, facecolors='none', 
                   edgecolors=marker_color, s=80, linewidth=2,
                   label='Trained on Emerging Markets Only'),
        plt.scatter([], [], marker=mixed_marker, facecolors=marker_color, 
                   edgecolors=marker_color, s=80, linewidth=2,
                   label='Trained on Developed and Emerging Markets'),
    ]
    
    # Place legend at the bottom
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
              ncol=2, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.15)
    
    # Add group boxes and labels AFTER layout adjustments
    for group_name, (start_idx, end_idx) in group_positions.items():
        if group_name != 'ungrouped' and start_idx != end_idx:  # Only for actual groups with multiple items
            # Calculate positions for the box in figure coordinates
            top_ax = axes[start_idx]
            bottom_ax = axes[end_idx]
            
            # Get the bbox of the subplots in figure coordinates AFTER layout adjustments
            top_bbox = top_ax.get_position()
            bottom_bbox = bottom_ax.get_position()
            
            # Box coordinates (with small padding)
            box_left = 0
            box_right = -0.005
            box_top = top_bbox.y1
            box_bottom = bottom_bbox.y0

            # Draw the box
            box = Rectangle((box_left, box_bottom), 
                          box_right - box_left, 
                          box_top - box_bottom,
                          linewidth=0.5, 
                          edgecolor='gray', 
                          facecolor='gray',
                          transform=fig.transFigure,
                          clip_on=False)
            fig.patches.append(box)
            
            # Add the group label on the left side
            label_x = box_left - 0.03
            label_y = (box_top + box_bottom) / 2
            fig.text(label_x, label_y, group_name, 
                    transform=fig.transFigure,
                    fontsize=14, ha='center', va='center', 
                    rotation=90)
    
    plt.show()