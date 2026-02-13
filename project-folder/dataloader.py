"""
Cricket Dataset Loader and Utilities
Handles cricket dataset downloading, loading, preprocessing, and CSV conversion.
"""

import os
import numpy as np
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class CricketDatasetUtilities:
    """
    Utility class for cricket dataset operations
    """
    
    @staticmethod
    def download_cricket_dataset():
        """Download the cricket dataset from Kaggle"""
        print("Downloading cricket dataset from Kaggle...")
        try:
            data_path = kagglehub.dataset_download("akarshsinghh/cricket-player-performance-prediction")
            print(f"Dataset downloaded to: {data_path}")
            
            # Check available files
            files = os.listdir(data_path)
            print(f"Files in dataset: {files}")
            
            csv_files = [f for f in files if f.endswith('.csv')]
            print(f"CSV files found: {csv_files}")
            
            return data_path
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            return None
    
    @staticmethod
    def load_and_preprocess_cricket_data(data_path=None):
        """Load and preprocess the cricket data"""
        print("Loading and preprocessing cricket data...")
        
        # Download data if path not provided
        if data_path is None:
            data_path = CricketDatasetUtilities.download_cricket_dataset()
            if data_path is None:
                return None, None, None
        
        # Load the match data
        match_file = os.path.join(data_path, 'match.csv')
        match_df = pd.read_csv(match_file)
        
        print(f"Original data shape: {match_df.shape}")
        print(f"Columns: {list(match_df.columns)}")
        
        # Drop unnecessary columns
        drop_cols = [
            'Unnamed: 0',        # index column
            'match detail id',   # pure identifier
            'scorecard id'       # pure identifier
        ]
        match_df = match_df.drop(columns=drop_cols, errors='ignore')
        
        # Handle date columns
        match_df['start_date'] = pd.to_datetime(match_df['start_date'], errors='coerce')
        match_df['match_year'] = match_df['start_date'].dt.year
        match_df['match_month'] = match_df['start_date'].dt.month
        
        # Drop the original date column after extracting features
        match_df = match_df.drop(columns=['start_date'], errors='ignore')
        
        # Encode categorical columns
        le = LabelEncoder()
        for col in match_df.select_dtypes(include='object').columns:
            match_df[col] = le.fit_transform(match_df[col])
        
        # Create target variable based on runs scored
        if 'match_id' in match_df.columns and 'runs' in match_df.columns:
            match_results = match_df.groupby('match_id')['runs'].transform('max')
            match_df['winner'] = (match_df['runs'] == match_results).astype(int)
        else:
            # Fallback: create a binary target based on runs being above median
            median_runs = match_df['runs'].median() if 'runs' in match_df.columns else 50
            match_df['winner'] = (match_df['runs'] > median_runs).astype(int)
        
        # Separate features and target
        X = match_df.drop('winner', axis=1)
        y = match_df['winner']
        
        print(f"Feature count: {X.shape[1]}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, match_df
    
    @staticmethod
    def save_dataset_to_csv(df, filename="cricket_dataset.csv"):
        """Save cricket dataset to CSV file"""
        try:
            df.to_csv(filename, index=False)
            print(f"Dataset successfully saved as: {filename}")
            print(f"Total Rows: {df.shape[0]}")
            print(f"Total Columns: {df.shape[1]}")
            return True
        except Exception as e:
            print(f"Error saving dataset: {str(e)}")
            return False
    
    @staticmethod
    def load_and_scale_cricket_data(data_path=None):
        """
        Load cricket dataset, preprocess, and return scaled train-test split
        """
        try:
            # Get preprocessed data
            X, y, match_df = CricketDatasetUtilities.load_and_preprocess_cricket_data(data_path)
            
            if X is None:
                return None, None, None, None, None, None
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"Training set shape: {X_train.shape}")
            print(f"Test set shape: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test, match_df, scaler
            
        except Exception as e:
            print(f"Error loading and scaling data: {str(e)}")
            return None, None, None, None, None, None
    
    @staticmethod
    def create_cricket_test_csv(test_size=0.2, filename="cricket_test.csv"):
        """Create a test CSV file for cricket data"""
        print("Creating cricket test dataset...")
        
        try:
            # Load and preprocess data
            X, y, match_df = CricketDatasetUtilities.load_and_preprocess_cricket_data()
            
            if X is None:
                print("Failed to load cricket data.")
                return False
            
            # Combine features and target
            complete_df = X.copy()
            complete_df['winner'] = y
            
            # Split into train and test
            train_df, test_df = train_test_split(
                complete_df, test_size=test_size, random_state=42, stratify=y
            )
            
            # Save test dataset
            test_df.to_csv(filename, index=False)
            print(f"Cricket test dataset saved as: {filename}")
            print(f"Test samples: {len(test_df)}")
            print(f"Features: {len(test_df.columns) - 1}")
            print(f"Target distribution: {test_df['winner'].value_counts().to_dict()}")
            
            # Also save full dataset
            full_filename = filename.replace('test', 'full')
            complete_df.to_csv(full_filename, index=False)
            print(f"Full cricket dataset saved as: {full_filename}")
            
            return True
            
        except Exception as e:
            print(f"Error creating test CSV: {str(e)}")
            return False


def load_dataset():
    """
    Main function to load cricket dataset - compatible with application.py
    Returns train-test split for compatibility with existing application code
    """
    X_train, X_test, y_train, y_test, _, _ = CricketDatasetUtilities.load_and_scale_cricket_data()
    return X_train, X_test, y_train, y_test


def main():
    """Main function for dataset operations"""
    print("Cricket Dataset Utilities")
    print("="*50)
    
    choice = input("""
Choose an option:
1. Download and preprocess cricket data
2. Create cricket test CSV file
3. Load and display data info
4. Exit

Enter your choice (1-4): """)
    
    if choice == "1":
        X, y, df = CricketDatasetUtilities.load_and_preprocess_cricket_data()
        if X is not None:
            print("\nDataset loaded successfully!")
            print(f"Features shape: {X.shape}")
            print(f"Target shape: {y.shape}")
            
            # Ask if user wants to save
            save_choice = input("\nSave dataset to CSV? (y/n): ").lower()
            if save_choice == 'y':
                complete_df = X.copy()
                complete_df['winner'] = y
                CricketDatasetUtilities.save_dataset_to_csv(complete_df, "cricket_dataset.csv")
        
    elif choice == "2":
        filename = input("Enter test CSV filename (default: cricket_test.csv): ").strip()
        if not filename:
            filename = "cricket_test.csv"
        CricketDatasetUtilities.create_cricket_test_csv(filename=filename)
        
    elif choice == "3":
        X_train, X_test, y_train, y_test, df, scaler = CricketDatasetUtilities.load_and_scale_cricket_data()
        if X_train is not None:
            print(f"\nDataset Information:")
            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            print(f"Features: {X_train.shape[1]}")
            print(f"Training target distribution: {y_train.value_counts().to_dict()}")
            print(f"Test target distribution: {y_test.value_counts().to_dict()}")
            
    elif choice == "4":
        print("Exiting...")
        
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()