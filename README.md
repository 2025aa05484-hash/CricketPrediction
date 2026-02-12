# Cricket Winning Prediction System

A machine learning application for predicting cricket match outcomes using various algorithms.

## Problem Statement

The goal of this project is to develop a machine learning system that can predict cricket match outcomes (win/loss) based on various match and player performance features. The system aims to assist cricket analysts, fans, and betting platforms in making informed decisions by leveraging historical cricket data and advanced machine learning algorithms.

## Dataset Description

The dataset used for this project is the "Cricket Player Performance Prediction" dataset from Kaggle, which contains comprehensive cricket match data with the following characteristics:

- **Dataset Size**: 6,199 records with 15 features
- **Data Source**: Kaggle dataset by akarshsinghh
- **Target Variable**: Binary classification (Winner/Not Winner)
- **Key Features**:
  - Match details (match_number, match_id, series_id)
  - Player information (name, team_id, opp_team_id)
  - Performance metrics (runs, over, run_rate)
  - Match metadata (start_date, matchtype, title)
- **Data Split**: 80% training (4,959 samples), 20% testing (1,240 samples)
- **Class Distribution**: Class 0: 3,305 samples, Class 1: 2,894 samples

## Models Used

## Models Used

The following machine learning models were implemented and evaluated for cricket match outcome prediction:

### Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.6911 | 0.7598 | 0.6725 | 0.6598 | 0.6661 | 0.3789 |
| Decision Tree | 0.6718 | 0.7094 | 0.6564 | 0.6235 | 0.6395 | 0.3390 |
| kNN | 0.6073 | 0.6445 | 0.5816 | 0.5665 | 0.5739 | 0.2099 |
| Naive Bayes | 0.6266 | 0.6916 | 0.5983 | 0.6097 | 0.6039 | 0.2509 |
| Random Forest (Ensemble) | 0.7177 | 0.7967 | 0.6998 | 0.6926 | 0.6962 | 0.4327 |
| XGBoost (Ensemble) | 0.7113 | 0.7971 | 0.6928 | 0.6857 | 0.6892 | 0.4197 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-----------------------------------|
| Logistic Regression | Shows good balanced performance with 69.11% accuracy. Demonstrates reliable linear classification capabilities with decent AUC (0.7598). Performs well for interpretable predictions with moderate complexity. |
| Decision Tree | Achieves 67.18% accuracy with good interpretability. The model shows clear decision boundaries but may suffer from overfitting. Lower MCC (0.3390) indicates moderate correlation between predictions and actual values. |
| kNN | Lowest performing model with 60.73% accuracy. The algorithm struggles with the high-dimensional feature space and shows poor generalization. Low AUC (0.6445) suggests difficulty in distinguishing between classes. |
| Naive Bayes | Moderate performance at 62.66% accuracy. The independence assumption of features limits its effectiveness on this dataset. Shows better recall than precision, indicating tendency to predict positive class more frequently. |
| Random Forest (Ensemble) | **Best overall performer** with 71.77% accuracy and highest F1-score (0.6962). The ensemble approach effectively reduces overfitting and provides robust predictions. Excellent balance between precision and recall. |
| XGBoost (Ensemble) | Second-best performer with 71.13% accuracy and **highest AUC (0.7971)**. Superior gradient boosting shows excellent probability calibration. Strong performance across all metrics with good generalization capability. |

## Features

- Multiple ML models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost
- Interactive web interface using Streamlit
- Data preprocessing and feature engineering
- Model performance evaluation and comparison
- Comprehensive test data predictions analysis
- Confusion matrix and classification reports
- Visualization of model performance comparisons

## Usage

### Method 1: Run the Complete Pipeline
1. Navigate to the project-folder directory:
   ```
   cd project-folder
   ```

2. Run the integrated pipeline:
   ```
   python main.py
   ```
   This will automatically:
   - Download the cricket dataset from Kaggle
   - Train all 6 models
   - Display performance metrics and test predictions
   - Generate visualization charts

### Method 2: Run the Streamlit Web Interface
1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Upload your cricket dataset (CSV format)
3. Select a machine learning model
4. Train the model and view predictions

## Project Structure

```
CricketPrediction/
├── app.py                        # Main Streamlit application
├── CricketWinningPrediction.ipynb # Jupyter notebook version
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── project-folder/
    ├── main.py                   # Integrated ML pipeline script
    ├── decision_tree.py          # Decision Tree model implementation
    ├── knn.py                    # K-Nearest Neighbors model
    ├── logistic.py               # Logistic Regression model
    ├── naive_bayes.py            # Naive Bayes model
    ├── random_forest.py          # Random Forest ensemble model
    └── xgboost_model.py          # XGBoost ensemble model
```

## Models

- **Logistic Regression**: Linear classification model for binary outcome prediction
- **Decision Tree**: Tree-based decision making with interpretable rules
- **KNN**: K-Nearest Neighbors classifier using distance-based classification
- **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
- **Random Forest**: Ensemble of decision trees for improved accuracy and robustness
- **XGBoost**: Gradient boosting framework for high-performance ensemble learning

## Results Summary

- **Best Overall Model**: Random Forest (F1-Score: 0.6962, Accuracy: 71.77%)
- **Best AUC Score**: XGBoost (AUC: 0.7971)
- **Most Interpretable**: Decision Tree (Accuracy: 67.18%)
- **Fastest Training**: Naive Bayes (Accuracy: 62.66%)

The ensemble methods (Random Forest and XGBoost) consistently outperformed individual algorithms, demonstrating the effectiveness of combining multiple learners for cricket match outcome prediction.

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv cricket_env
   cricket_env\Scripts\activate  # On Windows
   ```
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Install additional required packages:
   ```
   pip install kagglehub
   pip install scikit-learn --only-binary=all
   pip install joblib
   ```

### Alternative Installation (Individual Packages)
If you encounter issues with the requirements.txt, you can install packages individually:

```bash
# Core ML and Data Science packages
pip install streamlit==1.28.0
pip install pandas==2.1.0
pip install numpy==1.24.3
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install plotly==5.15.0

# Machine Learning packages
pip install scikit-learn --only-binary=all
pip install xgboost==2.0.0
pip install joblib
```

**Note**: We use `--only-binary=all` for scikit-learn to avoid compilation issues on Windows systems without Microsoft Visual C++ Build Tools.