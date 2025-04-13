# Enhancing Bank Consumer Retention via Churn Prediction

📊 Predicting customer churn in banking using ML/DL models, SMOTE for imbalance correction, and variable importance analysis.

## 📚 Project Summary
This project explores the application of supervised machine learning and deep learning models to predict customer churn in banks. The models are tested with and without the application of SMOTE for balancing imbalanced datasets.

- Dataset: Kaggle Bank Customer Churn Dataset (10,000 rows, 14 columns)
- Task: Binary classification (Churn: 1/0)
- Tools: EDA, model training, SMOTE application, feature importance analysis

## 🔧 Key Features
- SMOTE vs No-SMOTE predictive modeling
- ML models: Logistic Regression, Random Forest, LightGBM, XGBoost, AdaBoost, Extra Trees
- DL Model: Keras Neural Network with dropout regularization
- Performance Metrics: Accuracy, Precision, Recall, F1, AUC
- Variable importance using Random Forest

## 🧠 Models Used
- Logistic Regression
- Deep Learning (Keras)
- AdaBoost
- Extra Trees Classifier
- XGBoost
- LightGBM
- Random Forest

## 🧪 Results Snapshot (With SMOTE)
| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 65.84%   | 65.56%    | 66.76% | 66.16%   |
| Deep Learning      | 65.00%   | 52.86%    | 53.60% | 52.70%   |
| AdaBoost           | 81.35%   | 81.21%    | 81.58% | 81.39%   |
| Random Forest      | 84.76%   | 84.94%    | 84.51% | 84.73%   |
| XGBoost            | 84.95%   | 84.48%    | 85.64% | 85.06%   |
| Extra Trees        | 85.06%   | 84.91%    | 85.27% | 85.09%   |
| LightGBM           | **85.16%**| 85.06%    | 85.31% | **85.18%** |

> LightGBM and Random Forest offer the best performance, especially after SMOTE.

## 📁 Repository Layout
- `data/`: Raw and processed customer churn data
- `notebooks/`: Jupyter notebooks for EDA, preprocessing, training
- `src/`: Scripts for modeling, preprocessing, and evaluation
- `models/`: Trained model binaries
- `reports/`: Graphs and result summaries

## 📦 Requirements
```bash
pip install -r requirements.txt
Includes:
pandas, numpy, scikit-learn, xgboost, lightgbm, seaborn, matplotlib, imbalanced-learn, keras, tensorflow


