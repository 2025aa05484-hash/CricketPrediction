"""
Decision Tree Model for Cricket Winning Prediction
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

class DecisionTreeModel:
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5)
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.is_trained = False
        
    def preprocess_data(self, X, y=None):
        """Preprocess the data for training"""
        X_processed = X.copy()
        
        # Handle datetime columns
        datetime_cols = X_processed.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            X_processed[col + '_year'] = X_processed[col].dt.year
            X_processed[col + '_month'] = X_processed[col].dt.month
            X_processed[col + '_day'] = X_processed[col].dt.day
            X_processed = X_processed.drop(col, axis=1)
        
        # Handle categorical columns
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_processed[col] = X_processed[col].fillna('Unknown')
                X_processed[col] = self.label_encoders[col].fit_transform(X_processed[col].astype(str))
            else:
                X_processed[col] = X_processed[col].fillna('Unknown')
                # Handle unseen labels
                test_labels = X_processed[col].astype(str)
                encoded_labels = []
                for label in test_labels:
                    if label in self.label_encoders[col].classes_:
                        encoded_labels.append(self.label_encoders[col].transform([label])[0])
                    else:
                        encoded_labels.append(0)  # Default value for unseen labels
                X_processed[col] = encoded_labels
        
        # Handle missing values
        X_processed = pd.DataFrame(
            self.imputer.fit_transform(X_processed) if not self.is_trained else self.imputer.transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )
        
        return X_processed
    
    def train(self, X, y):
        """Train the decision tree model"""
        # Preprocess data
        X_processed = self.preprocess_data(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': dict(zip(X_processed.columns, self.model.feature_importances_))
        }
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_processed = self.preprocess_data(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_processed = self.preprocess_data(X)
        return self.model.predict_proba(X_processed)
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before accessing feature importance")
        
        return self.model.feature_importances_