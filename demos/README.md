## Notebook Demos

The Jupyter Notebooks in `demos` can be used to run the experiments described in the report. The Notebooks are best run sequentially as each experiment and analysis builds on the previous.

Here is a brief description of the content in each Notebook:

| Notebook | Description |
|----------|-------------|
| `0.Introduction.ipynb` | Sample plots to showcase crisis distributions and motivate the project. |
| `1.DataPreparation.ipynb` | Data loading, cleaning, and preprocessing pipeline. Handles missing values, outliers, and feature engineering for both developed and emerging markets datasets. |
| `2.BaselineClassifiers.ipynb` | Implementation and evaluation of tree-based machine learning classifiers as baseline models for crisis prediction. |
| `3.FeatureSelection.ipynb` | Exploration of different types of dimensionality reduction methods (e.g. PCA, t-SNE, UMAP, VAE) performed on the data.
| `4.MLVSDLClassifiers.ipynb` | Performance comparison of machine learning with linear and non-linear dimensionality reduction techniques. |
| `5.LSTMClassifier.ipynb` | End-to-end LSTM classifier with attention mechanisms, trained using precision-based focal loss. Includes transfer learning experiments for emerging markets. |
| `6.FeatureSpace.ipynb` | A visualization of the feature space of the trained LSTM Classifier. |
