"""Cricket Dataset Utilities
Data loading, preprocessing, and CSV handling utilities for cricket prediction models.
"""

import pandas as pd
import numpy as np
import kagglehub
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os


class CricketDatasetUtilities:
    """Utilities for cricket dataset operations including download, preprocessing, and CSV operations."""
    
    def __init__(self):
        self.dataset_path = None
        self.raw_data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def download_cricket_dataset(self):
        """Download cricket dataset from Kaggle."""
        try:
            # Download the dataset using kagglehub
            self.dataset_path = kagglehub.dataset_download("bhavikjikadara/cricket-player-performance-prediction")
            return self.dataset_path
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    
    def load_and_preprocess_cricket_data(self):
        """Load and preprocess cricket dataset for ML models."""
        try:
            if not self.dataset_path:
                self.download_cricket_dataset()
            
            # Find CSV file in dataset directory
            csv_file = None
            for file in os.listdir(self.dataset_path):
                if file.endswith('.csv'):
                    csv_file = os.path.join(self.dataset_path, file)
                    break
            
            if not csv_file:
                raise FileNotFoundError("No CSV file found in dataset")
            
            # Load the dataset
            self.raw_data = pd.read_csv(csv_file)
            
            # Basic preprocessing
            df = self.raw_data.copy()
            
            # Handle missing values
            df = df.dropna()
            
            # Remove any obvious non-feature columns
            columns_to_drop = ['ID', 'id', 'Name', 'name'] if any(col in df.columns for col in ['ID', 'id', 'Name', 'name']) else []
            if columns_to_drop:
                df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            
            # Encode categorical variables
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = self.label_encoder.fit_transform(df[col])
            
            self.processed_data = df
            return df
            
        except Exception as e:
            print(f"Error loading and preprocessing data: {e}")
            return None
    
    def save_dataset_to_csv(self, filename="cricket_dataset.csv"):
        """Save processed dataset to CSV file."""
        try:
            if self.processed_data is not None:
                self.processed_data.to_csv(filename, index=False)
                return filename
            else:
                print("No processed data available to save")
                return None
        except Exception as e:
            print(f"Error saving dataset: {e}")
            return None
    
    def load_and_scale_cricket_data(self):
        """Load and scale cricket data for ML models."""
        try:
            if self.processed_data is None:
                self.load_and_preprocess_cricket_data()
            
            if self.processed_data is None:
                return None, None
            
            # Assume last column is target
            X = self.processed_data.iloc[:, :-1]
            y = self.processed_data.iloc[:, -1]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            
            return X_scaled, y
            
        except Exception as e:
            print(f"Error scaling data: {e}")
            return None, None
    
    def create_cricket_test_csv(self, X_test, y_test, model_name="general"):
        """Create CSV file from test data."""
        try:
            # Combine X_test and y_test
            if hasattr(X_test, 'columns'):
                # X_test is a DataFrame
                test_df = X_test.copy()
            else:
                # X_test is numpy array
                test_df = pd.DataFrame(X_test)
            
            # Add target column
            if hasattr(y_test, 'name') and y_test.name:
                target_col_name = y_test.name
            else:
                target_col_name = 'target'
            
            test_df[target_col_name] = y_test
            
            # Save to CSV
            filename = f"cricket_test_split_{model_name}.csv"
            test_df.to_csv(filename, index=False)
            return filename
            
        except Exception as e:
            print(f"Error creating test CSV: {e}")
            return None


def load_dataset():
    """Convenient function to load cricket dataset."""
    utilities = CricketDatasetUtilities()
    return utilities.load_and_preprocess_cricket_data()


# For backward compatibility and direct script execution
if __name__ == "__main__":
    # Example usage
    utilities = CricketDatasetUtilities()
    
    # Download and process dataset
    data = utilities.load_and_preprocess_cricket_data()
    
    if data is not None:
        print("Dataset loaded successfully!")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Save to CSV
        csv_file = utilities.save_dataset_to_csv()
        if csv_file:
            print(f"Dataset saved to: {csv_file}")
    else:
        print("Failed to load dataset")