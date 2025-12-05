# Car_Project
Car Price Prediction using Machine Learning

ML Models — Regression & Logistic Regression on Used-Car Datasets

📌 Project Overview

This project applies Machine Learning techniques to analyze car features and predict their selling price.
Two public datasets from Kaggle are used to train and evaluate different models:

Dataset	Purpose
Used Car Price Prediction	Primary dataset for price regression
Indian Cars Dataset	Secondary dataset for dataset comparison & generalization

🔗 Dataset Sources:

https://www.kaggle.com/datasets/vrajesh0sharma7/used-car-price-prediction

https://www.kaggle.com/datasets/medhekarabhinav5/indian-cars-dataset

🎯 Goals of the Project

✔ Predict market price of used cars using regression models
✔ Compare performance of multiple ML algorithms
✔ Convert the problem into a classification task (price category) for logistic regression
✔ Analyze the effects of car features on its selling price

📂 Project Structure
📁 Car-Price-Prediction-ML
│── data/                       → CSV datasets (not uploaded due to size)
│── notebooks/                  → Jupyter notebooks for development
│── src/
│     ├── preprocessing.py       → Data cleaning and feature engineering
│     ├── regression_models.py   → Linear / Random Forest / Gradient Boosting
│     ├── classification_models.py → Logistic Regression & other classifiers
│     ├── utils.py               → Helper functions
│── results/
│     ├── model_scores.csv       → Evaluation metrics
│     ├── feature_importance.png → Visualizations
│── README.md
│── requirements.txt
│── main.py                      → Main executable script

🔧 Tech Stack
Category	Tools
Language	Python
Data Handling	Pandas, NumPy
Visualization	Matplotlib, Seaborn
ML Algorithms	Scikit-learn
Notebook Dev	Jupyter Notebook
🧠 Models Implemented
🔹 Regression Models

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

🔹 Classification Models (Price Category Prediction)

Logistic Regression

Random Forest Classifier

Decision Tree Classifier

🧪 Evaluation Metrics
Model Type	Metrics Used
Regression	RMSE, MAE, R² Score
Classification	Accuracy, Precision, Recall, F1-Score, Confusion Matrix
