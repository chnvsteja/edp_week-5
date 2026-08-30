# Week 5 – Random Forest / XGBoost – Train Classifier

## 📌 Overview

Week 5 focuses on learning and implementing two important Machine Learning classification algorithms: **Random Forest** and **XGBoost**.

The main goal of this week is to train classifiers for the **Credit Card Fraud Detection** project. The training data prepared during Week 4 is used to train the Machine Learning models.

The trained models learn patterns from previous credit card transactions and can later be used to classify transactions as either **legitimate** or **fraudulent**.

## 🎯 Week 5 Goal

**Learn:** Random Forest / XGBoost

**Build:** Train Classifier

The main objective is to successfully train Machine Learning classifiers using the prepared and balanced training data.

## 🧠 Classification

Classification is a type of supervised Machine Learning where a model learns from labelled data and predicts the category of new data.

In this project, the problem is a **binary classification problem** because there are two possible classes:

- `0` → Legitimate transaction
- `1` → Fraudulent transaction

The model learns the relationship between transaction features and their corresponding class labels.

## 📊 Data Preparation

The data preparation was completed during Week 4.

The credit card transaction dataset was loaded and divided into input features and the target variable.

The input features are stored in `X`, while the target variable is stored in `y`.

The dataset was then divided into training and testing data.

The training data was used to teach the Machine Learning models, while the test data was kept separate for future evaluation.

## ⚠️ Class Imbalance

Credit card fraud datasets are highly imbalanced because legitimate transactions are much more common than fraudulent transactions.

For example, the number of legitimate transactions can be much larger than the number of fraudulent transactions.

This imbalance can make it difficult for a Machine Learning model to properly learn fraud patterns.

To address this problem, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data.

After applying SMOTE, the training data contained a more balanced number of legitimate and fraudulent transactions.

The balanced training data used in Week 5 is:

```python
X_train_smote
y_train_smote
