# Multi-class Classification

This repository contains a comprehensive analysis and implementation of Multi-class Classification strategies using Logistic Regression. The project focuses on predicting obesity levels based on eating habits and physical condition, utilizing both a Jupyter Notebook for detailed analysis and a generalized Streamlit web application for interactive modeling.

## 📄 Overview

Multi-class classification is a classification task with more than two classes; each sample may only be labeled as one class. This project demonstrates how to handle such problems using Logistic Regression by employing strategies like **One-vs-All (OvA/OvR)** and **One-vs-One (OvO)**.

The project utilizes the **Obesity Risk Prediction dataset** (`Obesity_level_prediction_dataset.csv`), which includes data on individual attributes, eating habits, and physical condition to classify individuals into various weight categories ranging from Insufficient Weight to Obesity Type III.

## 📂 Repository Structure

* **`Multi-class_Classification.ipynb`**: A Jupyter Notebook containing the step-by-step data analysis, preprocessing (One-Hot Encoding, Scaling), and model implementation using Scikit-Learn. It compares OvA and OvO strategies.
* **`app.py`**: A generalized **Streamlit** web application. It allows users to upload *any* dataset (CSV), perform Exploratory Data Analysis (EDA), preprocess data, train a Logistic Regression classifier, and generate predictions on new data.
* **`Obesity_level_prediction_dataset.csv`**: The dataset used for the analysis, sourced from the UCI Machine Learning Repository.
* **`requirements.txt`**: A list of Python dependencies required to run the project.
* **`LICENSE`**: The MIT License governing the use of this software.

## 📊 Dataset Details

The dataset consists of 2,111 records and 17 attributes. The target variable is **`NObeyesdad`**, which categorizes obesity levels into 7 distinct classes:

* Insufficient\_Weight
* Normal\_Weight
* Overweight\_Level\_I
* Overweight\_Level\_II
* Obesity\_Type\_I
* Obesity\_Type\_II
* Obesity\_Type\_III

**Key Attributes:**
* **Physical:** Gender, Age, Height, Weight.
* **Eating Habits:** FAVC (High caloric food), FCVC (Vegetables), NCP (Main meals), CAEC (Food between meals), CH2O (Water consumption), CALC (Alcohol).
* **Physical Condition:** SCC (Calorie monitoring), FAF (Physical activity frequency), TUE (Time using technology devices), MTRANS (Transportation used).

## 🛠 Methodology

### 1. Data Preprocessing
* **Feature Scaling:** Continuous numerical variables (e.g., Age, Height, Weight) are standardized using `StandardScaler` to improve model convergence.
* **Encoding:** Categorical variables (e.g., Gender, MTRANS) are transformed using `OneHotEncoder` to convert them into a numerical format suitable for Logistic Regression.

### 2. Classification Strategies
The project implements Logistic Regression using two primary strategies for multi-class problems:
* **One-vs-All (OvA) / One-vs-Rest (OvR):** Trains **K** binary classifiers (where **K** is the number of classes). Each classifier predicts whether an instance belongs to a specific class or not.
* **One-vs-One (OvO):** Trains **K(K-1)/2** binary classifiers for every pair of classes. The class with the most votes is chosen as the prediction.

### 3. Web Application (`app.py`)
The Streamlit app provides a generalized GUI wrapper for the classification pipeline:
* **Dynamic Inputs:** Users can select Target and Feature columns from any uploaded CSV.
* **Preprocessing:** Options to toggle Standardization and One-Hot Encoding.
* **Visualization:** Interactive correlation heatmaps, distribution plots, and confusion matrices.
* **Prediction:** Ability to upload new batch data or manually input values to predict classes using the trained model.

## 🚀 How to Run

### Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment.

### Installation
1.  Clone the repository or download the files.
2.  Install the required dependencies listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Notebook
To view the detailed analysis and code walkthrough:
```bash
jupyter notebook Multi-class_Classification.ipynb
```

### Running the Streamlit App
To launch the interactive web application:
```bash
streamlit run app.py
```
Once running, the app will open in your default web browser. You can upload `Obesity_level_prediction_dataset.csv` to test the functionality.

## 📈 Results
The analysis in the notebook highlights:
* The effectiveness of One-Hot Encoding for categorical rich datasets.
* The trade-offs between One-vs-Rest (faster training) and One-vs-One (potentially higher accuracy but computationally expensive) strategies.
* The evaluation metrics (Accuracy, Confusion Matrix) used to determine the model's ability to distinguish between similar classes (e.g., Obesity Type I vs Type II).

## 📜 License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🤝 Acknowledgments
* **Dataset Source**: UCI Machine Learning Repository & palechor, de la Hoz-Manotas (2019). "Dataset for estimation of obesity levels based on eating habits and physical condition".
* **Libraries**: Scikit-Learn, Pandas, NumPy, Streamlit, Matplotlib, Seaborn.