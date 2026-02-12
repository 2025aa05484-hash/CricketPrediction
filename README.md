# Cricket Winning Prediction System

A machine learning application for predicting cricket match outcomes using various algorithms.

## Features

- Multiple ML models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost
- Interactive web interface using Streamlit
- Data preprocessing and feature engineering
- Model performance evaluation and comparison

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Upload your cricket dataset (CSV format)
3. Select a machine learning model
4. Train the model and view predictions

## Project Structure

```
project-folder/
│── app.py                 # Main Streamlit application
│── requirements.txt       # Python dependencies
│── README.md             # This file
│── model/                # ML model implementations
│     ├── logistic.py
│     ├── decision_tree.py
│     ├── knn.py
│     ├── naive_bayes.py
│     ├── random_forest.py
│     └── xgboost.py
```

## Models

- **Logistic Regression**: Linear classification model
- **Decision Tree**: Tree-based decision making
- **KNN**: K-Nearest Neighbors classifier
- **Naive Bayes**: Probabilistic classifier
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting framework

## Contributing

Feel free to contribute by adding new models or improving existing functionality.