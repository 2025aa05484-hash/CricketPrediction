"""
Cricket Winning Prediction App
Main application file for the cricket prediction system.
"""

import streamlit as st
import pandas as pd
import numpy as np
from model.logistic import LogisticModel
from model.decision_tree import DecisionTreeModel
from model.knn import KNNModel
from model.naive_bayes import NaiveBayesModel
from model.random_forest import RandomForestModel
from model.xgboost import XGBoostModel

def main():
    st.title("Cricket Winning Prediction System")
    st.sidebar.title("Model Selection")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
    )
    
    st.write(f"Selected Model: {model_choice}")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())
        
        if st.button("Train Model"):
            # Initialize the selected model
            model_map = {
                "Logistic Regression": LogisticModel(),
                "Decision Tree": DecisionTreeModel(),
                "KNN": KNNModel(),
                "Naive Bayes": NaiveBayesModel(),
                "Random Forest": RandomForestModel(),
                "XGBoost": XGBoostModel()
            }
            
            model = model_map[model_choice]
            st.success(f"{model_choice} model initialized!")

if __name__ == "__main__":
    main()