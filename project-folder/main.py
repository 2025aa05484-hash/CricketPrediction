"""
Cricket Player Performance Prediction - Integrated Script
Converts the Jupyter notebook functionality into a single Python script
that uses the modular model classes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# Import custom model classes
from decision_tree import DecisionTreeModel
from knn import KNNModel
from logistic import LogisticModel
from naive_bayes import NaiveBayesModel
from random_forest import RandomForestModel
from xgboost_model import XGBoostModel


class CricketPredictionPipeline:
    """
    Main pipeline class for cricket performance prediction
    """
    
    def __init__(self):
        self.data_path = None
        self.match_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def download_dataset(self):
        """Download the cricket dataset from Kaggle"""
        print("Downloading cricket dataset from Kaggle...")
        try:
            self.data_path = kagglehub.dataset_download("akarshsinghh/cricket-player-performance-prediction")
            print(f"Dataset downloaded to: {self.data_path}")
            
            # Check available files
            files = os.listdir(self.data_path)
            print(f"Files in dataset: {files}")
            
            csv_files = [f for f in files if f.endswith('.csv')]
            print(f"CSV files found: {csv_files}")
            
            return True
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            return False
    
    def load_and_preprocess_data(self):
        """Load and preprocess the cricket data"""
        print("Loading and preprocessing data...")
        
        # Load the match data
        match_file = os.path.join(self.data_path, 'match.csv')
        self.match_df = pd.read_csv(match_file)
        
        print(f"Original data shape: {self.match_df.shape}")
        print(f"Columns: {list(self.match_df.columns)}")
        
        # Drop unnecessary columns
        drop_cols = [
            'Unnamed: 0',        # index column
            'match detail id',   # pure identifier
            'scorecard id'       # pure identifier
        ]
        self.match_df = self.match_df.drop(columns=drop_cols, errors='ignore')
        
        # Handle date columns
        self.match_df['start_date'] = pd.to_datetime(self.match_df['start_date'], errors='coerce')
        self.match_df['match_year'] = self.match_df['start_date'].dt.year
        self.match_df['match_month'] = self.match_df['start_date'].dt.month
        
        # Drop the original date column after extracting features
        self.match_df = self.match_df.drop(columns=['start_date'], errors='ignore')
        
        # Encode categorical columns initially
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        
        for col in self.match_df.select_dtypes(include='object').columns:
            self.match_df[col] = le.fit_transform(self.match_df[col])
        
        # Create target variable based on runs scored
        # Group by match_id and determine winner based on highest runs
        if 'match_id' in self.match_df.columns and 'runs' in self.match_df.columns:
            match_results = self.match_df.groupby('match_id')['runs'].transform('max')
            self.match_df['winner'] = (self.match_df['runs'] == match_results).astype(int)
        else:
            # Fallback: create a binary target based on runs being above median
            median_runs = self.match_df['runs'].median() if 'runs' in self.match_df.columns else 50
            self.match_df['winner'] = (self.match_df['runs'] > median_runs).astype(int)
        
        # Separate features and target
        X = self.match_df.drop('winner', axis=1)
        y = self.match_df['winner']
        
        print(f"Feature count: {X.shape[1]}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def split_and_scale_data(self, X, y):
        """Split data into train/test sets and scale features"""
        print("Splitting and scaling data...")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        # Note: Individual models will handle their own preprocessing
        # This allows each model to apply its specific requirements
        
        return True
    
    def initialize_models(self):
        """Initialize all model instances"""
        print("Initializing models...")
        
        self.models = {
            "Decision Tree": DecisionTreeModel(),
            "KNN": KNNModel(n_neighbors=5),
            "Logistic Regression": LogisticModel(),
            "Naive Bayes": NaiveBayesModel(),
            "Random Forest": RandomForestModel(n_estimators=100, max_depth=10),
            "XGBoost": XGBoostModel(n_estimators=100, max_depth=6, learning_rate=0.1)
        }
        
        print(f"Initialized {len(self.models)} models")
    
    def train_models(self):
        """Train all models and collect results"""
        print("Training models...")
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            try:
                # Train the model (model handles its own preprocessing)
                training_results = model.train(self.X_train, self.y_train)
                
                # Get predictions on test set
                y_pred = model.predict(self.X_test)
                y_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate evaluation metrics
                metrics = {
                    "Accuracy": accuracy_score(self.y_test, y_pred),
                    "Precision": precision_score(self.y_test, y_pred, zero_division=0),
                    "Recall": recall_score(self.y_test, y_pred, zero_division=0),
                    "F1": f1_score(self.y_test, y_pred, zero_division=0),
                    "MCC": matthews_corrcoef(self.y_test, y_pred)
                }
                
                if y_prob is not None:
                    metrics["AUC"] = roc_auc_score(self.y_test, y_prob)
                else:
                    metrics["AUC"] = 0.0
                
                self.results[name] = metrics
                print(f"{name} trained successfully! Accuracy: {metrics['Accuracy']:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        print(f"\nTraining completed! {len(self.results)} models trained successfully.")
    
    def display_results(self):
        """Display model performance results"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE RESULTS")
        print("="*60)
        
        # Convert results to DataFrame for better visualization
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        
        print(results_df.to_string())
        
        # Find the best performing model for each metric
        print("\n" + "-"*40)
        print("BEST MODELS BY METRIC:")
        print("-"*40)
        for metric in results_df.columns:
            best_model = results_df[metric].idxmax()
            best_score = results_df[metric].max()
            print(f"{metric:10s}: {best_model} ({best_score:.4f})")
        
        # Overall best model based on F1 score (balanced metric)
        best_overall = results_df['F1'].idxmax()
        print(f"\nBest Overall Model (by F1-Score): {best_overall}")
        print(f"F1-Score: {results_df.loc[best_overall, 'F1']:.4f}")
        
        return results_df
    
    def create_visualizations(self, results_df):
        """Create and save model comparison visualizations"""
        print("\nCreating visualizations...")
        
        plt.figure(figsize=(15, 10))
        
        # Plot accuracy for all models
        plt.subplot(2, 3, 1)
        results_df['Accuracy'].plot(kind='bar', color='skyblue')
        plt.title('Model Accuracy Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Plot F1 Score
        plt.subplot(2, 3, 2)
        results_df['F1'].plot(kind='bar', color='lightgreen')
        plt.title('Model F1-Score Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('F1-Score')
        plt.grid(True, alpha=0.3)
        
        # Plot AUC
        plt.subplot(2, 3, 3)
        results_df['AUC'].plot(kind='bar', color='lightcoral')
        plt.title('Model AUC Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('AUC')
        plt.grid(True, alpha=0.3)
        
        # Plot MCC
        plt.subplot(2, 3, 4)
        results_df['MCC'].plot(kind='bar', color='lightsalmon')
        plt.title('Model MCC Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('MCC')
        plt.grid(True, alpha=0.3)
        
        # Plot Precision
        plt.subplot(2, 3, 5)
        results_df['Precision'].plot(kind='bar', color='lightpink')
        plt.title('Model Precision Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Precision')
        plt.grid(True, alpha=0.3)
        
        # Plot Recall
        plt.subplot(2, 3, 6)
        results_df['Recall'].plot(kind='bar', color='lightblue')
        plt.title('Model Recall Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Recall')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = 'cricket_model_comparison.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Visualization saved as '{plot_filename}'")
        
        # Also create a summary metrics heatmap
        plt.figure(figsize=(10, 6))
        import seaborn as sns
        sns.heatmap(results_df.T, annot=True, cmap='YlOrRd', fmt='.3f', cbar=True)
        plt.title('Model Performance Heatmap')
        plt.xlabel('Models')
        plt.ylabel('Metrics')
        plt.xticks(rotation=45, ha='right')
        
        heatmap_filename = 'cricket_model_heatmap.png'
        plt.tight_layout()
        plt.savefig(heatmap_filename, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved as '{heatmap_filename}'")
        
        plt.close('all')  # Close all figures to free memory
    
    def run_complete_pipeline(self):
        """Run the complete machine learning pipeline"""
        print("Starting Cricket Performance Prediction Pipeline...")
        print("="*60)
        
        # Step 1: Download dataset
        if not self.download_dataset():
            print("Failed to download dataset. Exiting.")
            return False
        
        # Step 2: Load and preprocess data
        X, y = self.load_and_preprocess_data()
        
        # Step 3: Split and scale data
        if not self.split_and_scale_data(X, y):
            print("Failed to split data. Exiting.")
            return False
        
        # Step 4: Initialize models
        self.initialize_models()
        
        # Step 5: Train models
        self.train_models()
        
        if not self.results:
            print("No models were trained successfully. Exiting.")
            return False
        
        # Step 6: Display results
        results_df = self.display_results()
        
        # Step 7: Create visualizations
        self.create_visualizations(results_df)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return True
    
    def predict_new_data(self, new_data, model_name="Random Forest"):
        """Make predictions on new data using a specific model"""
        if model_name not in self.models:
            print(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
            return None
        
        model = self.models[model_name]
        if not model.is_trained:
            print(f"Model '{model_name}' is not trained yet.")
            return None
        
        try:
            predictions = model.predict(new_data)
            probabilities = model.predict_proba(new_data) if hasattr(model, 'predict_proba') else None
            
            return {
                'predictions': predictions,
                'probabilities': probabilities
            }
        except Exception as e:
            print(f"Error making predictions with {model_name}: {str(e)}")
            return None


def main():
    """Main function to run the cricket prediction pipeline"""
    # Create and run the pipeline
    pipeline = CricketPredictionPipeline()
    
    try:
        success = pipeline.run_complete_pipeline()
        
        if success:
            print("\nPipeline completed successfully!")
            
            # Example of how to use the trained models for predictions
            print("\nExample: To make predictions on new data:")
            print("prediction_result = pipeline.predict_new_data(new_data_df, 'Random Forest')")
            
        else:
            print("Pipeline failed to complete.")
            
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()