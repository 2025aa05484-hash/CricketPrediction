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
    
    @staticmethod
    def download_cricket_dataset():
        """Download cricket dataset from Kaggle."""
        try:
            # Download the dataset using kagglehub
            dataset_path = kagglehub.dataset_download("bhavikjikadara/cricket-player-performance-prediction")
            return dataset_path
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    
    @staticmethod
    def load_and_preprocess_cricket_data():
        """Load and preprocess cricket dataset for ML models."""
        try:
            # Download dataset
            dataset_path = CricketDatasetUtilities.download_cricket_dataset()
            if not dataset_path:
                raise Exception("Failed to download dataset")
            
            # Find CSV file in dataset directory
            csv_file = None
            for file in os.listdir(dataset_path):
                if file.endswith('.csv'):
                    csv_file = os.path.join(dataset_path, file)
                    break
            
            if not csv_file:
                raise FileNotFoundError("No CSV file found in dataset")
            
            # Load the dataset
            df = pd.read_csv(csv_file)
            
            # Basic preprocessing
            # Handle missing values
            df = df.dropna()
            
            # Remove any obvious non-feature columns
            columns_to_drop = []
            for col in ['ID', 'id', 'Name', 'name', 'Player_Name', 'player_name']:
                if col in df.columns:
                    columns_to_drop.append(col)
            
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
            
            # Encode categorical variables
            le = LabelEncoder()
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = le.fit_transform(df[col])
            
            # Assume the target column is 'winner' or the last column
            if 'winner' in df.columns:
                target_col = 'winner'
            elif 'target' in df.columns:
                target_col = 'target'
            else:
                target_col = df.columns[-1]  # Use last column as target
            
            # Prepare features and target
            X = df.drop(target_col, axis=1)
            y = df[target_col]
            
            return X, y, df
            
        except Exception as e:
            print(f"Error loading and preprocessing data: {e}")
            return None, None, None
    
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
                X, y, df = self.load_and_preprocess_cricket_data()
                self.processed_data = df
            else:
                # Assume last column is target
                X = self.processed_data.iloc[:, :-1]
                y = self.processed_data.iloc[:, -1]
            
            if X is None:
                return None, None
            
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
            if hasattr(X_test, 'copy'):
                test_df = X_test.copy()
            else:
                test_df = pd.DataFrame(X_test)
            
            # Add target column
            if hasattr(y_test, 'name') and y_test.name:
                target_col_name = y_test.name
            else:
                target_col_name = 'winner'
            
            test_df[target_col_name] = y_test
            
            # Save to CSV
            filename = f"cricket_test_split_{model_name}.csv"
            test_df.to_csv(filename, index=False)
            return filename
            
        except Exception as e:
            print(f"Error creating test CSV: {e}")
            return None


def load_dataset():
    """Convenient function to load cricket dataset - compatible with app.py"""
    try:
        # Use static method to load data
        X, y, df = CricketDatasetUtilities.load_and_preprocess_cricket_data()
        
        if X is not None and y is not None:
            # Split into train/test for compatibility
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            return X_train, X_test, y_train, y_test
        else:
            return None, None, None, None
    
    except Exception as e:
        print(f"Error in load_dataset: {e}")
        return None, None, None, None


# For backward compatibility and direct script execution
if __name__ == "__main__":
    # Example usage
    print("Loading cricket dataset...")
    
    # Test the load function
    X_train, X_test, y_train, y_test = load_dataset()
    
    if X_train is not None:
        print("Dataset loaded successfully!")
        print(f"Training shape: {X_train.shape}")
        print(f"Test shape: {X_test.shape}")
        print(f"Target distribution: {y_train.value_counts().to_dict()}")
        
        # Test utilities class
        utilities = CricketDatasetUtilities()
        csv_file = utilities.create_cricket_test_csv(X_test, y_test, "sample")
        if csv_file:
            print(f"Test CSV created: {csv_file}")
    else:
        print("Failed to load dataset")