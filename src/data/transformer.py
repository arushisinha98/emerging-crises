import pandas as pd
import numpy as np
from sklearn.utils import resample

class DummyEncode:
    """
    A class for encoding categorical variables into dummy variables with proper handling
    of unseen categories during transform.
    """
    
    def __init__(self, column_name):
        """
        Initialize the DummyEncode class.
        
        Parameters:
        column_name (str): Name of the categorical column to encode
        """
        self.column_name = column_name
        self.categories_ = None
        self.dummy_columns_ = None
        
    def fit(self, df):
        """
        Fit the encoder on the training data to learn the categories.
        
        Parameters:
        df (pd.DataFrame): Training dataframe containing the categorical column
        
        Returns:
        pd.DataFrame: Dataframe with original columns plus dummy columns
        """
        if self.column_name not in df.columns:
            raise ValueError(f"Column '{self.column_name}' not found in dataframe")
        
        # Store unique categories seen during training
        self.categories_ = sorted(df[self.column_name].unique())
        
        # Create dummy column names
        self.dummy_columns_ = [f"{self.column_name}_{cat}" for cat in self.categories_]
        
        # Create dummy variables
        dummies = pd.get_dummies(df[self.column_name], prefix=self.column_name)
        
        # Ensure all expected columns are present (in case some categories are missing in this batch)
        for col in self.dummy_columns_:
            if col not in dummies.columns:
                dummies[col] = 0
                
        # Reorder columns to match expected order
        dummies = dummies[self.dummy_columns_]
        
        # Combine with original dataframe
        result_df = df.copy()
        result_df = pd.concat([result_df, dummies], axis=1)
        
        return result_df
    
    def transform(self, df):
        """
        Transform new data using the fitted encoder.
        
        Parameters:
        df (pd.DataFrame): Dataframe to transform
        
        Returns:
        pd.DataFrame: Dataframe with original columns plus dummy columns
        """
        if self.categories_ is None:
            raise ValueError("Encoder must be fitted before transform. Call fit() first.")
            
        if self.column_name not in df.columns:
            raise ValueError(f"Column '{self.column_name}' not found in dataframe")
        
        # Create dummy variables for the data
        dummies = pd.get_dummies(df[self.column_name], prefix=self.column_name)
        
        # Create a dataframe with all expected dummy columns initialized to 0
        dummy_df = pd.DataFrame(0, index=df.index, columns=self.dummy_columns_)
        
        # Fill in the values for categories that exist in both training and current data
        for col in dummies.columns:
            if col in dummy_df.columns:
                dummy_df[col] = dummies[col]
        
        # Combine with original dataframe
        result_df = df.copy()
        result_df = pd.concat([result_df, dummy_df], axis=1)
        
        return result_df
    
    def fit_transform(self, df):
        """
        Fit the encoder and transform the data in one step.
        
        Parameters:
        df (pd.DataFrame): Training dataframe to fit and transform
        
        Returns:
        pd.DataFrame: Transformed dataframe with dummy columns
        """
        return self.fit(df)
    
class DownsampleMajority:
    """
    A class for downsampling the majority class to create a balanced dataset
    """
    def __init__(self, random_state=42):
        """
        Initialize the DownsampleMajority class.

        Parameters:
        df (pd.DataFrame): The input dataframe
        labels (pd.Series): The labels corresponding to the dataframe
        """
        self.random_state=random_state

    @staticmethod
    def downsample(df, labels, random_state=42):
        majority = resample(df[labels == 0], replace=False,
                            n_samples=len(df[labels == 1]),
                            random_state=random_state)
        minority = df[labels == 1]
        downsample_df = pd.concat([majority, minority])
        downsample_labels = np.asarray([np.zeros(majority.shape[0]), np.ones(minority.shape[0])]).flatten()
        return downsample_df, downsample_labels
    
    def transform(self, df, labels):
        """
        Transform data using the fitted downsampler.
        """
        downsample_df, downsample_labels = self.downsample(df, labels, self.random_state)
        return downsample_df, downsample_labels
