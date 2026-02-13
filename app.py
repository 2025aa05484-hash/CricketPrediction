"""Cricket Winning Prediction App - Complete ML Pipeline
Streamlit web application for cricket match outcome prediction using multiple ML models.
Comprehensive training, evaluation, and analysis system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project-folder to path to import model classes
project_folder_path = os.path.join(os.path.dirname(__file__), 'project-folder')
sys.path.append(project_folder_path)

# Import ML models and utilities  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Import dataset utilities from main.py
try:
    from main import CricketDatasetUtilities, load_dataset
except ImportError as e:
    st.error(f"Error importing from main.py: {e}")
    st.stop()

# Import custom model classes
try:
    from decision_tree import DecisionTreeModel
    from knn import KNNModel
    from logistic import LogisticModel
    from naive_bayes import NaiveBayesModel
    from random_forest import RandomForestModel
    from xgboost_model import XGBoostModel
except ImportError as e:
    st.error(f"Error importing model classes: {e}")
    st.stop()

# Model Registry - Cricket Classification Models
MODEL_REGISTRY = {
    "Logistic Regression": LogisticModel,
    "Decision Tree": DecisionTreeModel, 
    "KNN": KNNModel,
    "Naive Bayes": NaiveBayesModel,
    "Random Forest": RandomForestModel,
    "XGBoost": XGBoostModel
}

# Test file for cricket dataset
LOCAL_TEST_FILE = "cricket_test.csv"

@st.cache_data
def load_cricket_dataset():
    """Load and preprocess cricket dataset using utilities from main.py"""
    try:
        # Use the utilities from main.py
        X, y, match_df = CricketDatasetUtilities.load_and_preprocess_cricket_data()
        
        if X is None:
            st.error("Failed to load cricket dataset.")
            return None, None, None
        
        return X, y, match_df
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, None, None

def train_all_models(X_train, y_train, X_test, y_test):
    """Train all models and return comprehensive results like main.py"""
    models = {
        "Decision Tree": DecisionTreeModel(),
        "KNN": KNNModel(n_neighbors=5),
        "Logistic Regression": LogisticModel(),
        "Naive Bayes": NaiveBayesModel(),
        "Random Forest": RandomForestModel(n_estimators=100, max_depth=10),
        "XGBoost": XGBoostModel(n_estimators=100, max_depth=6, learning_rate=0.1)
    }
    
    results = {}
    predictions = {}
    training_progress = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        training_progress.text(f"Training {name}... ({i+1}/{len(models)})")
        
        try:
            # Train the model
            model.train(X_train, y_train)
            
            # Get predictions on test set
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Store predictions for later analysis
            predictions[name] = {
                'y_pred': y_pred,
                'y_prob': y_prob,
                'y_true': y_test,
                'model': model
            }
            
            # Calculate evaluation metrics
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0),
                "MCC": matthews_corrcoef(y_test, y_pred)
            }
            
            if y_prob is not None:
                metrics["AUC"] = roc_auc_score(y_test, y_prob)
            else:
                metrics["AUC"] = 0.0
            
            results[name] = metrics
            
        except Exception as e:
            st.warning(f"Error training {name}: {str(e)}")
            continue
    
    training_progress.empty()
    return results, predictions

def create_comprehensive_visualizations(results_df):
    """Create comprehensive visualizations like main.py"""
    
    # Create subplots using plotly
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=['Accuracy', 'F1-Score', 'AUC', 'MCC', 'Precision', 'Recall'],
        specs=[[{"secondary_y": False}]*3, [{"secondary_y": False}]*3]
    )
    
    metrics = ['Accuracy', 'F1', 'AUC', 'MCC', 'Precision', 'Recall']
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink', 'lightblue']
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        row, col = positions[i]
        
        fig.add_trace(
            go.Bar(
                x=results_df.index,
                y=results_df[metric],
                name=metric,
                marker_color=color,
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title_text="Cricket Model Performance Comparison",
        height=800,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=results_df.T.values,
        x=results_df.index,
        y=results_df.columns,
        colorscale='YlOrRd',
        text=results_df.T.round(3).values,
        texttemplate="%{text}",
        textfont={"size":10},
        hoverongaps=False
    ))
    
    fig_heatmap.update_layout(
        title="Model Performance Heatmap",
        xaxis_title="Models",
        yaxis_title="Metrics"
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

def display_comprehensive_predictions_analysis(results, predictions):
    """Display comprehensive test predictions analysis like main.py"""
    if not predictions:
        st.warning("No predictions available.")
        return
    
    # Get the best performing model
    best_model_name = max(results.items(), key=lambda x: x[1]['F1'])[0]
    
    st.success(f"🏆 **Best Performing Model:** {best_model_name}")
    st.write(f"**F1-Score:** {results[best_model_name]['F1']:.4f}")
    
    # Show sample predictions from the best model
    y_pred = predictions[best_model_name]['y_pred']
    y_true = predictions[best_model_name]['y_true']
    y_prob = predictions[best_model_name]['y_prob']
    
    # Sample predictions table
    st.subheader("🔍 Sample Test Predictions (First 15 samples)")
    
    sample_size = min(15, len(y_true))
    sample_data = []
    
    for i in range(sample_size):
        actual = y_true.iloc[i] if hasattr(y_true, 'iloc') else y_true[i]
        predicted = y_pred[i]
        prob = y_prob[i] if y_prob is not None else None
        correct = "✅" if actual == predicted else "❌"
        
        sample_data.append({
            "Index": i,
            "Actual": actual,
            "Predicted": predicted,
            "Probability": f"{prob:.3f}" if prob is not None else "N/A",
            "Correct": correct
        })
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True)
    
    # Confusion Matrix for best model
    st.subheader(f"🎯 Confusion Matrix - {best_model_name}")
    
    cm = confusion_matrix(y_true, y_pred)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        cm_df = pd.DataFrame(
            cm,
            columns=["Predicted: Loss (0)", "Predicted: Win (1)"],
            index=["Actual: Loss (0)", "Actual: Win (1)"]
        )
        st.dataframe(cm_df)
        
        # Confusion matrix visualization
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Loss (0)", "Win (1)"],
            y=["Loss (0)", "Win (1)"],
            color_continuous_scale="Blues",
            text_auto=True
        )
        fig_cm.update_layout(title=f"Confusion Matrix - {best_model_name}")
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.write("**Confusion Matrix Interpretation:**")
        st.write(f"True Negatives (Correct Loss): {cm[0,0]}")
        st.write(f"False Positives (Wrong Win): {cm[0,1]}")
        st.write(f"False Negatives (Wrong Loss): {cm[1,0]}")
        st.write(f"True Positives (Correct Win): {cm[1,1]}")
        
        # Additional metrics
        total_samples = len(y_true)
        correct_predictions = cm[0,0] + cm[1,1]
        accuracy = correct_predictions / total_samples
        error_rate = (total_samples - correct_predictions) / total_samples
        
        st.metric("Total Test Samples", total_samples)
        st.metric("Correct Predictions", correct_predictions)
        st.metric("Accuracy", f"{accuracy:.3f}")
        st.metric("Error Rate", f"{error_rate:.3f}")
    
    # Classification Report
    st.subheader("📋 Detailed Classification Report")
    class_report = classification_report(
        y_true, y_pred,
        target_names=['Loss (0)', 'Win (1)'],
        output_dict=True
    )
    
    report_df = pd.DataFrame(class_report).transpose()
    st.dataframe(report_df, use_container_width=True)
    
    # Model Comparison Summary
    st.subheader("📊 All Models Test Set Results Summary")
    
    comparison_data = []
    for model_name in predictions:
        y_pred_model = predictions[model_name]['y_pred']
        y_true_model = predictions[model_name]['y_true']
        
        correct = np.sum(y_true_model == y_pred_model)
        incorrect = len(y_true_model) - correct
        accuracy = correct / len(y_true_model)
        error_rate = incorrect / len(y_true_model)
        
        comparison_data.append({
            "Model": model_name,
            "Correct": correct,
            "Incorrect": incorrect,
            "Accuracy": f"{accuracy:.3f}",
            "Error Rate": f"{error_rate:.3f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Prediction Distribution Analysis
    st.subheader("📈 Prediction Distribution Analysis")
    
    distribution_data = []
    for model_name in predictions:
        y_pred_model = predictions[model_name]['y_pred']
        class_0_pred = np.sum(y_pred_model == 0)
        class_1_pred = np.sum(y_pred_model == 1)
        
        distribution_data.append({
            "Model": model_name,
            "Class 0 (Loss)": class_0_pred,
            "Class 1 (Win)": class_1_pred
        })
    
    # Add actual distribution
    class_0_actual = np.sum(y_true == 0)
    class_1_actual = np.sum(y_true == 1)
    distribution_data.append({
        "Model": "Actual Distribution",
        "Class 0 (Loss)": class_0_actual,
        "Class 1 (Win)": class_1_actual
    })
    
    distribution_df = pd.DataFrame(distribution_data)
    st.dataframe(distribution_df, use_container_width=True)

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return metrics"""
    try:
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0), 
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, y_pred)
        }
        
        # Add AUC if model supports probability prediction
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            results["AUC"] = roc_auc_score(y_test, y_prob)
        else:
            results["AUC"] = 0.0
            
        return results, y_pred
    
    except Exception as e:
        st.error(f"Error evaluating model: {str(e)}")
        return None, None

def main():
    st.set_page_config(page_title="Cricket Winning Prediction System", layout="wide")
    st.title("🏏 Cricket Winning Prediction System")
    st.markdown("**Comprehensive Machine Learning Pipeline for Cricket Match Outcome Prediction**")
    
    # Sidebar configuration
    st.sidebar.title("🔧 Configuration Panel")
    
    # Mode selection
    app_mode = st.sidebar.selectbox(
        "Choose Application Mode:",
        ["Single Model Evaluation", "Complete Pipeline Training", "CSV File Evaluation"],
        help="Select the mode for cricket prediction analysis"
    )
    
    if app_mode == "CSV File Evaluation":
        # -------- Download Sample Test File (FIRST) --------
        st.subheader("📁 Cricket Dataset Evaluation")
        st.markdown("Upload a cricket test dataset to evaluate trained models")
        
        # Check if we have a local test file
        if os.path.exists(LOCAL_TEST_FILE):
            with open(LOCAL_TEST_FILE, "rb") as f:
                st.download_button(
                    label="Download Sample Cricket Test File (cricket_test.csv)",
                    data=f,
                    file_name=LOCAL_TEST_FILE,
                    mime="text/csv"
                )
        else:
            st.info("Create a sample test file using main.py for download")

        # -------- Upload CSV File --------
        st.subheader("📤 Upload Test Dataset (.csv only)")
        uploaded_file = st.file_uploader(
            "Upload CSV file (Test Data Only)",
            type=["csv"],
            help="CSV should contain cricket match features with 'winner' column as target"
        )

        # -------- Model Selection --------
        st.subheader("🤖 Choose the Model")
        selected_model_name = st.selectbox(
            "Select Model",
            list(MODEL_REGISTRY.keys()),
            help="Choose the machine learning model for evaluation"
        )

        # -------- Run Evaluation --------
        if st.button("🚀 Run Model Evaluation", type="primary"):

            with st.spinner("Loading data and evaluating model..."):

                # Load Test Data
                if uploaded_file is not None:
                    test_df = pd.read_csv(uploaded_file)
                    st.info("Using uploaded test dataset.")
                else:
                    if not os.path.exists(LOCAL_TEST_FILE):
                        st.error("No uploaded file and default test file not found. Please upload a CSV file or create one using main.py")
                        return
                    test_df = pd.read_csv(LOCAL_TEST_FILE)
                    st.info("Using default cricket_test.csv from local folder.")

                # Check for target column
                if "winner" not in test_df.columns:
                    st.error("CSV must contain a 'winner' column as target variable.")
                    return

                X_test = test_df.drop("winner", axis=1)
                y_test = test_df["winner"]
                
                st.write(f"**Test Data Shape:** {X_test.shape}")
                st.write(f"**Target Distribution:** {y_test.value_counts().to_dict()}")

                # Train Model on full cricket dataset
                try:
                    # Load training data
                    X_train, _, y_train, _ = load_dataset()  # From main.py
                    
                    if X_train is None:
                        st.error("Failed to load training data. Check main.py utilities.")
                        return
                    
                    # Initialize and train model
                    model_class = MODEL_REGISTRY[selected_model_name]
                    model = model_class()
                    
                    with st.spinner(f"Training {selected_model_name}..."):
                        model.train(X_train, y_train)
                    
                    st.success(f"✅ {selected_model_name} trained successfully!")
                    
                    # Evaluate model
                    results, y_pred = evaluate_model(model, X_test, y_test)
                    
                    if results is None:
                        st.error("Model evaluation failed.")
                        return

                except Exception as e:
                    st.error(f"Error during training/evaluation: {str(e)}")
                    return

            # -------- Display Results --------
            st.success("🎉 Model evaluation completed successfully!")

            # Metrics display
            st.subheader("📊 Evaluation Metrics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{results['Accuracy']:.4f}")
                st.metric("Precision", f"{results['Precision']:.4f}")
            with col2:
                st.metric("Recall", f"{results['Recall']:.4f}")
                st.metric("F1 Score", f"{results['F1 Score']:.4f}")
            with col3:
                st.metric("AUC", f"{results['AUC']:.4f}" if results['AUC'] else "N/A")
                st.metric("MCC", f"{results['MCC']:.4f}")

            # Detailed metrics table
            metrics_df = pd.DataFrame(list(results.items()), columns=["Metric", "Value"])
            st.dataframe(metrics_df, use_container_width=True)

            # Confusion Matrix
            st.subheader("🎯 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            col1, col2 = st.columns(2)
            
            with col1:
                cm_df = pd.DataFrame(
                    cm,
                    columns=["Predicted: Loss (0)", "Predicted: Win (1)"],
                    index=["Actual: Loss (0)", "Actual: Win (1)"]
                )
                st.dataframe(cm_df)
            
            with col2:
                # Confusion matrix visualization
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Loss (0)", "Win (1)"],
                    y=["Loss (0)", "Win (1)"],
                    color_continuous_scale="Blues",
                    text_auto=True,
                    title="Confusion Matrix"
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            
            # Sample predictions
            st.subheader("🔍 Sample Predictions")
            display_sample_predictions(y_test, y_pred, model, X_test, n_samples=15)
    
    else:
        # Original comprehensive training modes
        handle_comprehensive_training_modes(app_mode)


def display_sample_predictions(y_test, y_pred, model, X_test, n_samples=10):
    """Display sample predictions with probabilities"""
    sample_size = min(n_samples, len(y_test))
    
    sample_data = []
    for i in range(sample_size):
        actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        predicted = y_pred[i]
        correct = "✅" if actual == predicted else "❌"
        
        sample_row = {
            "Index": i,
            "Actual": actual,
            "Predicted": predicted,
            "Correct": correct
        }
        
        # Add probability if available
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            prob = y_prob[i, 1] if len(y_prob.shape) > 1 else y_prob[i]
            sample_row["Probability"] = f"{prob:.3f}"
        else:
            sample_row["Probability"] = "N/A"
            
        sample_data.append(sample_row)
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True)

def handle_comprehensive_training_modes(app_mode):
    """Handle the comprehensive training modes (original functionality)"""
    
    # Data loading options
    st.sidebar.subheader("📊 Data Configuration")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Auto-load Kaggle Dataset", "Upload Custom CSV"],
        help="Auto-load uses the cricket dataset from Kaggle"
    )
    
    # Training parameters
    st.sidebar.subheader("🎯 Training Parameters")
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20, 5) / 100
    random_state = st.sidebar.number_input("Random State", value=42, min_value=0)
    
    # Initialize data variables
    X, y, df = None, None, None
    
    # Handle data loading
    if data_source == "Auto-load Kaggle Dataset":
        st.subheader("📥 Loading Cricket Dataset from Kaggle")
        
        with st.spinner("Downloading and preprocessing cricket dataset..."):
            X, y, df = load_cricket_dataset()
        
        if X is not None:
            st.success("✅ Dataset loaded successfully!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Features", X.shape[1])
            with col3:
                st.metric("Win Class", int((y == 1).sum()))
            with col4:
                st.metric("Loss Class", int((y == 0).sum()))
            
            # Show data preview
            with st.expander("📋 Dataset Overview", expanded=False):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Features (X) - First 10 rows:**")
                    st.dataframe(X.head(10))
                    
                    st.write("**Feature Statistics:**")
                    st.dataframe(X.describe())
                
                with col2:
                    st.write("**Target Distribution (y):**")
                    target_counts = y.value_counts()
                    
                    fig = px.bar(
                        x=target_counts.index,
                        y=target_counts.values,
                        labels={'x': 'Class', 'y': 'Count'},
                        title='Target Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("**Target Statistics:**")
                    st.write(f"Class 0 (Loss): {target_counts[0]} ({target_counts[0]/len(y)*100:.1f}%)")
                    st.write(f"Class 1 (Win): {target_counts[1]} ({target_counts[1]/len(y)*100:.1f}%)")
        
    elif data_source == "Upload Custom CSV":
        st.subheader("📁 Upload Custom Cricket Dataset")
        uploaded_file = st.file_uploader(
            "Upload CSV file", 
            type=['csv'],
            help="CSV should contain cricket match data with a 'winner' column as target"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
                
                # Check if winner column exists
                if 'winner' not in df.columns:
                    st.error("CSV must contain a 'winner' column as target variable.")
                    return
                
                # Prepare data
                X = df.drop('winner', axis=1)
                y = df['winner']
                
                # Show basic info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Samples", len(df))
                with col2:
                    st.metric("Features", X.shape[1])
                with col3:
                    st.metric("Win Class", int((y == 1).sum()))
                with col4:
                    st.metric("Loss Class", int((y == 0).sum()))
                
                # Show data preview
                with st.expander("📋 Dataset Overview", expanded=False):
                    st.dataframe(df.head())
                    
            except Exception as e:
                st.error(f"Error loading uploaded file: {str(e)}")
                return
    
    # Main application logic based on mode
    if X is not None and y is not None:
        
        if app_mode == "Single Model Training":
            st.subheader("🎯 Single Model Training & Evaluation")
            
            # Model selection
            model_choice = st.selectbox(
                "Choose a Machine Learning Model:",
                list(MODEL_REGISTRY.keys()),
                help="Select the ML algorithm for cricket match prediction"
            )
            
            # Model descriptions
            model_descriptions = {
                "Logistic Regression": "Linear classification model for binary outcome prediction",
                "Decision Tree": "Tree-based decision making with interpretable rules",
                "KNN": "K-Nearest Neighbors classifier using distance-based classification",
                "Naive Bayes": "Probabilistic classifier based on Bayes' theorem",
                "Random Forest": "Ensemble of decision trees for improved accuracy and robustness",
                "XGBoost": "Gradient boosting framework for high-performance ensemble learning"
            }
            
            st.info(f"**About {model_choice}:** {model_descriptions[model_choice]}")
            
            if st.button("🚀 Train Single Model", type="primary"):
                
                with st.spinner(f"Training {model_choice}..."):
                    try:
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state, stratify=y
                        )
                        
                        st.write(f"**Training samples:** {len(X_train)} | **Test samples:** {len(X_test)}")
                        
                        # Initialize and train model
                        model_class = MODEL_REGISTRY[model_choice]
                        model = model_class()
                        model.train(X_train, y_train)
                        
                        # Evaluate model
                        results, y_pred = evaluate_model(model, X_test, y_test)
                        
                        if results is not None:
                            st.success(f"✅ {model_choice} trained successfully!")
                            
                            # Display metrics
                            st.subheader("📊 Model Performance")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Accuracy", f"{results['Accuracy']:.4f}")
                                st.metric("Precision", f"{results['Precision']:.4f}")
                            
                            with col2:
                                st.metric("Recall", f"{results['Recall']:.4f}")
                                st.metric("F1 Score", f"{results['F1 Score']:.4f}")
                            
                            with col3:
                                st.metric("AUC", f"{results['AUC']:.4f}")
                                st.metric("MCC", f"{results['MCC']:.4f}")
                            
                            # Additional analysis sections
                            display_single_model_analysis(model, X_test, y_test, y_pred, results, model_choice)
                            
                    except Exception as e:
                        st.error(f"Error during model training: {str(e)}")
        
        elif app_mode == "Complete Pipeline Training":
            st.subheader("🏆 Complete ML Pipeline - All Models Comparison")
            st.markdown("Train and compare all 6 machine learning models simultaneously")
            
            if st.button("🚀 Train All Models & Compare", type="primary", key="train_all"):
                with st.spinner("Training all models... This may take a few moments."):
                    
                    try:
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state, stratify=y
                        )
                        
                        st.write(f"**Training samples:** {len(X_train)} | **Test samples:** {len(X_test)}")
                        
                        # Train all models
                        results, predictions = train_all_models(X_train, y_train, X_test, y_test)
                        
                        if results:
                            st.success(f"✅ Successfully trained {len(results)} models!")
                            
                            # Create comprehensive results DataFrame
                            results_df = pd.DataFrame(results).T
                            results_df = results_df.round(4)
                            
                            # Display results summary
                            st.subheader("📊 Model Performance Comparison")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Best models by metric
                            st.subheader("🏅 Best Models by Metric")
                            best_models_data = []
                            for metric in results_df.columns:
                                best_model = results_df[metric].idxmax()
                                best_score = results_df[metric].max()
                                best_models_data.append({
                                    "Metric": metric,
                                    "Best Model": best_model,
                                    "Score": f"{best_score:.4f}"
                                })
                            
                            best_models_df = pd.DataFrame(best_models_data)
                            st.dataframe(best_models_df, use_container_width=True)
                            
                            # Visualizations
                            st.subheader("📈 Performance Visualizations")
                            create_comprehensive_visualizations(results_df)
                            
                            # Comprehensive analysis
                            st.subheader("🔬 Detailed Predictions Analysis")
                            display_comprehensive_predictions_analysis(results, predictions)
                            
                        else:
                            st.error("❌ No models were trained successfully.")
                            
                    except Exception as e:
                        st.error(f"Error during pipeline execution: {str(e)}")
        
        else:  # Model Analysis mode
            st.subheader("🔬 Advanced Model Analysis")
            st.markdown("Detailed analysis and comparison tools")
            
            analysis_type = st.selectbox(
                "Choose Analysis Type:",
                ["Model Comparison", "Feature Importance", "Learning Curves", "Hyperparameter Analysis"]
            )
            
            if analysis_type == "Model Comparison":
                st.info("Select 'Complete Pipeline Training' mode for comprehensive model comparison.")
            
            else:
                st.info(f"{analysis_type} analysis coming soon!")

# Helper function to add footer
def add_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: grey; font-size: 14px;">
            Created by <b>Cricket ML Team</b> | Cricket Performance Prediction System
        </div>
        """,
        unsafe_allow_html=True
    )

def display_single_model_analysis(model, X_test, y_test, y_pred, results, model_name):
    """Display detailed analysis for single model"""
    
    # Confusion Matrix
    st.subheader("🎯 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        cm_df = pd.DataFrame(
            cm,
            columns=["Predicted: Loss (0)", "Predicted: Win (1)"],
            index=["Actual: Loss (0)", "Actual: Win (1)"]
        )
        st.dataframe(cm_df)
    
    with col2:
        st.write("**Confusion Matrix Interpretation:**")
        st.write(f"True Negatives (Correct Loss): {cm[0,0]}")
        st.write(f"False Positives (Wrong Win): {cm[0,1]}")
        st.write(f"False Negatives (Wrong Loss): {cm[1,0]}")
        st.write(f"True Positives (Correct Win): {cm[1,1]}")
    
    # Classification Report
    st.subheader("📋 Classification Report")
    class_report = classification_report(
        y_test, y_pred, 
        target_names=['Loss (0)', 'Win (1)'],
        output_dict=True
    )
    
    report_df = pd.DataFrame(class_report).transpose()
    st.dataframe(report_df, use_container_width=True)
    
    # Sample predictions
    st.subheader("🔍 Sample Predictions")
    display_sample_predictions(y_test, y_pred, model, X_test, n_samples=10)


if __name__ == "__main__":
    main()
    add_footer()