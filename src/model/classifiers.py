import numpy as np
import matplotlib.pyplot as plt
from typing import List

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

def plot_metrics(model, X_test, y_test):
    """
    Plot confusion matrix and ROC curve for model evaluation.
    
    Args:
        model: Trained model (sklearn or DeepLearningPipeline).
        X_test: Features for the test set (numpy array/pandas DataFrame for sklearn, 
                or DataFrame for DeepLearningPipeline).
        y_test: True labels for the test set.
    """
    # sequential model case
    if hasattr(model, 'predict_with_indices') and hasattr(model, 'get_aligned_labels'):
        y_pred, y_pred_proba, _ = model.predict_with_indices(X_test)
        # get positive class probabilities if both are returned
        y_pred_proba = y_pred_proba[:, 1] if len(y_pred_proba.shape) > 1 else y_pred_proba
        # get_aligned_labels for y_true
        # (len(y_true) <= len(y_test))
        y_true = model.get_aligned_labels(X_test, y_test)

    else:
        # standard model case
        if y_test is None:
            raise ValueError("y_test is required for sklearn models")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        # get positive class probabilities if both are returned
        y_pred_proba = y_pred_proba[:, 1] if len(y_pred_proba.shape) > 1 else y_pred_proba
        # (len(y_true) == len(y_test))
        y_true = y_test
    
    plt.figure(figsize=(10, 5))
        
    # Plot Confusion Matrix
    plt.subplot(1, 2, 1)
    plt.title('Confusion Matrix (%)')
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent)
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), values_format='.3f')
    disp.im_.set_clim(vmin=0, vmax=1)
    
    plt.subplot(1, 2, 2)
    plt.title('ROC Curve')
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_feature_importances(model,
                             feature_names: List[str], top_n: int=25,
                             feature_groups: dict[str, List[str]]=dict(),
                             title: str="Feature Importances"):
    """
    Plot feature importances from the Random Forest model.

    Parameters:
    ------------
    model:
        Trained sklearn model.
    feature_names: List
        Feature names.
    title: str
        Title for the plot.
    """
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    features = [feature_names[i] for i in indices]

    # Create a color palette for feature groups
    colors = []
    edgecolors = []
    linewidths = []
    hatches = []
    
    if feature_groups:
        group_names = list(feature_groups.keys())
        base_colors = [plt.cm.tab20(ii) for ii in range(len(group_names))]
        
        for feature in features:
            found = False
            for group_idx, (group, columns) in enumerate(feature_groups.items()):
                if feature in columns:
                    base_color = base_colors[group_idx]
                    if group_idx % 2 == 0:
                        # Even groups: filled bars with no outline
                        colors.append(base_color)
                        edgecolors.append('none')
                        linewidths.append(0)
                        hatches.append('')
                    else:
                        # Odd groups: hatch-style bars
                        colors.append('white')
                        edgecolors.append(base_color)
                        linewidths.append(1)
                        hatches.append('///')
                    found = True
                    break
            if not found:
                colors.append('grey')
                edgecolors.append('none')
                linewidths.append(0)
                hatches.append('')
    
    if not colors:
        colors = ['grey'] * top_n
        edgecolors = ['none'] * top_n
        linewidths = [0] * top_n
        hatches = [''] * top_n
    
    plt.figure(figsize=(10, 5))
    plt.title(title, fontsize=14)
    
    # Create bars with individual styling
    bars = plt.bar(range(top_n), importances[indices], align='center')
    for i, (bar, color, edgecolor, linewidth, hatch) in enumerate(zip(bars, colors, edgecolors, linewidths, hatches)):
        bar.set_facecolor(color)
        bar.set_edgecolor(edgecolor)
        bar.set_linewidth(linewidth)
        bar.set_hatch(hatch)

    plt.xticks(range(top_n), features, rotation=45, ha='right')
    plt.ylabel('Importance', fontsize=12)

    # Add legend if feature_groups is provided
    if feature_groups:
        legend_elements = []
        group_names = list(feature_groups.keys())
        base_colors = [plt.cm.tab20(ii) for ii in range(len(group_names))]
        
        for group_idx, group in enumerate(group_names):
            base_color = base_colors[group_idx]
            if group_idx % 2 == 0:
                # Even groups: filled with no outline
                legend_elements.append(plt.Rectangle((0,0),1,1, color=base_color, 
                                                   edgecolor='none', linewidth=0, label=group))
            else:
                # Odd groups: hatch-style
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='white', 
                                                   edgecolor=base_color, linewidth=1, hatch='///', label=group))
        
        plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

    return features

def stratified_gridsearch(param_grid: dict,
                          base_model,
                          X_train, y_train,
                          scoring='f1_weighted') -> GridSearchCV:
    
    X, _, y, _ = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
    cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv_folds, scoring=scoring, verbose=1
        )
    grid_search.fit(X, y)
    return grid_search