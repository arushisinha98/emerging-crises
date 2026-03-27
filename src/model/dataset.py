import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler

class BasicDataset(Dataset):
    """Dataset for basic dataset with features and targets"""
    def __init__(self, data, target):
        self.X = torch.tensor(data, dtype=torch.float32)
        self.y = torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]
    
    def get_input_dim(self):
        return self.X.shape[1]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SequentialDataset(Dataset):
    """Dataset for financial crisis prediction with multi-index dataframe"""
    
    def __init__(self, train_df, y_train, sequence_length=12, scaler=None, fit_scaler=True):
        """
        Initialize the dataset
        
        Args:
            train_df: pandas DataFrame
            y_train: numpy array of labels
            sequence_length: int, number of time steps in each sequence
            scaler: StandardScaler object (optional)
            fit_scaler: bool, whether to fit the scaler
        """
        self.sequence_length = sequence_length
        self.y_train = y_train

        # Get feature columns (exclude target)
        self.feature_cols = [col for col in train_df.columns if col not in ['Country', 'Date']]
        self.n_features = len(self.feature_cols)
        
        # Prepare sequences
        self.sequences = []
        self.labels = []
        self.countries = []
        self.valid_indices = []  # Track original indices for each sequence
        
        train_df_copy = train_df.copy()
        # Replace inf with NaN then forward fill only for numeric columns
        numeric_cols = train_df_copy.select_dtypes(include=[np.number]).columns
        train_df_copy[numeric_cols] = train_df_copy[numeric_cols].replace([np.inf, -np.inf], np.nan)
        train_df_copy[numeric_cols] = train_df_copy[numeric_cols].ffill()
        train_df_copy[numeric_cols] = train_df_copy[numeric_cols].fillna(0)

        # Process each country separately
        global_index = 0  # Track the global index across all countries
        for country in train_df_copy['Country'].unique():
            country_data = train_df_copy[train_df_copy['Country'] == country].sort_values(['Country', 'Date'])
            
            # Create sequences for this country
            for i in range(len(country_data) - sequence_length + 1):
                # Check if the sequence dates are consecutive
                sequence_dates = country_data['Date'].iloc[i:i+sequence_length]
                if not self._are_dates_consecutive(sequence_dates):
                    continue  # Skip this sequence if dates are not consecutive
                
                # Features sequence
                X_sequence = country_data[self.feature_cols].iloc[i:i+sequence_length].values
                # Target (label for the last time step in sequence)
                sequence_target_idx = country_data.index[i + sequence_length - 1]
                y_label = self.y_train[sequence_target_idx]

                self.sequences.append(X_sequence)
                self.labels.append(y_label)
                self.countries.append(country)
                self.valid_indices.append(sequence_target_idx)  # Store the original index
            
            global_index += len(country_data)
        
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)
        self.countries = np.array(self.countries)
        self.valid_indices = np.array(self.valid_indices)  # Convert to numpy array
        
        # Scale features
        if scaler is None:
            self.scaler = RobustScaler()
        else:
            self.scaler = scaler
            
        if fit_scaler:
            # Reshape for scaling (samples * timesteps, features)
            sequences_reshaped = self.sequences.reshape(-1, self.n_features)
            self.sequences_scaled = self.scaler.fit_transform(sequences_reshaped)
        else:
            sequences_reshaped = self.sequences.reshape(-1, self.n_features)
            self.sequences_scaled = self.scaler.transform(sequences_reshaped)
        
        # Reshape back to sequences
        self.sequences_scaled = self.sequences_scaled.reshape(-1, sequence_length, self.n_features)

    def _are_dates_consecutive(self, dates):
        """
        Check if a series of dates are consecutive (monthly or quarterly)
        
        Args:
            dates: pandas Series of dates
            
        Returns:
            bool: True if dates are consecutive, False otherwise
        """
        if len(dates) < 2:
            return True
            
        dates_sorted = dates.sort_values()
        
        # Calculate differences between consecutive dates
        date_diffs = dates_sorted.diff().dropna()
        if len(date_diffs) == 0:
            return True
        
        # Check if all differences are the same and within a tolerance
        mode_diff = date_diffs.mode().iloc[0]
        mode_diff_days = mode_diff.days

        if mode_diff_days == 1:
            tolerance = pd.Timedelta(days=1)  # Daily data
        elif 28 <= mode_diff_days <= 31:
            tolerance = pd.Timedelta(days=3)  # Monthly data
        elif 89 <= mode_diff_days <= 92:
            tolerance = pd.Timedelta(days=3)  # Quarterly data
        elif 365 <= mode_diff_days <= 366:
            tolerance = pd.Timedelta(days=1)  # Yearly data
        else:
            # Default tolerance for other frequencies (e.g., weekly)
            tolerance = pd.Timedelta(days=max(1, mode_diff_days // 10))

        return all(abs(diff - mode_diff) <= tolerance for diff in date_diffs)

    def __len__(self):
        """Return the total number of samples"""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            tuple: (input_tensor, target_tensor)
                - input_tensor: Tensor of shape (sequence_length, n_features)
                - target_tensor: Tensor containing the label (0 or 1)
        """
        sequence = torch.tensor(self.sequences_scaled[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return sequence, label
    
    def get_countries(self):
        """Return country labels for geographic cross-validation"""
        return self.countries
    
    def get_valid_indices(self):
        """Return the original DataFrame indices that correspond to valid sequences"""
        return self.valid_indices