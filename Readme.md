Customer Churn Prediction
1. Project Overview

This project aims to predict customer churn using machine learning. Customer churn refers to a situation in which a customer decides to stop using a company's services or subscription.

The goal is to train a machine learning model that can predict whether a customer will leave or remain a customer based on their characteristics and behavior.

We use a dataset containing 500 fictional customers and train a RandomForestClassifier model to predict customer churn.

2. Project Objectives

The main objectives of this project are:

Understand the basic concepts of machine learning.
Prepare and explore a structured dataset.
Identify the features used to predict customer churn.
Separate the dataset into training and testing sets.
Train a RandomForestClassifier model.
Evaluate the model using accuracy.
Save the trained model using joblib for future use.
3. Dataset Description

The dataset contains information about 500 fictional customers.

Features

The model uses the following input features:

Feature	Description
age	Customer's age
salaire	Customer's salary
abonnement	Customer's subscription information
anciennete	Customer's length of time with the company
consommation	Customer's consumption level
Target Variable

The target variable is churn.

Value	Meaning
0	The customer stays
1	The customer leaves

The model learns patterns from the input features to predict the target variable.

Note: The dataset contains fictional data created for educational purposes and does not represent real customers.
4. Key Machine Learning Concepts
Concept	Definition	Role in This Project
Dataset	A structured collection of data	The table of 500 fictional customers
Feature	An input variable used for prediction	Age, salary, subscription, seniority, and consumption
Target	The variable the model must predict	churn
Train/Test Split	Dividing data into training and testing sets	Evaluating how well the model generalizes to unseen data
Model	An algorithm that learns patterns from data	RandomForestClassifier
Hyperparameter	A setting defined before model training	n_estimators and max_depth
Accuracy	The proportion of correct predictions	The first metric used to evaluate the model
Serialization	Saving a Python object to disk	Saving the trained model as model.pkl using joblib
5. Machine Learning Workflow

The project follows these main steps:

Data preparation: Load and inspect the customer dataset.
Feature selection: Select the input variables and target variable.
Train/test split: Divide the dataset into training and testing sets.
Model creation: Initialize a RandomForestClassifier.
Model training: Train the model using the training data.
Model evaluation: Predict churn on the test data and calculate accuracy.
Model serialization: Save the trained model using joblib.
6. Model Used

We use the RandomForestClassifier algorithm from Scikit-learn.

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions. It can learn complex relationships between customer characteristics and churn.

Two important hyperparameters are:

n_estimators: The number of decision trees in the forest.
max_depth: The maximum depth allowed for each decision tree.

These parameters can be adjusted to experiment with the model's performance.

7. Model Evaluation

The first evaluation metric used in this project is accuracy.

Accuracy measures the proportion of correct predictions among all predictions made on the test set.

The formula is:

Accuracy
=
Correct Predictions
Total Predictions

For example, an accuracy of 80% means that the model correctly predicted 80% of the customers in the test set.

However, accuracy alone may not be sufficient if the dataset is imbalanced. Additional metrics, such as precision, recall, F1-score, and a confusion matrix, can provide a more complete evaluation.

8. Saving the Trained Model

After training, the model can be saved to a file using joblib.

Example:

import joblib

# Save the trained model
joblib.dump(model, "model.pkl")


The saved model can later be loaded without retraining it:

import joblib

# Load the trained model
model = joblib.load("model.pkl")


The variable model represents the trained classifier. The code assumes that the model has already been trained before saving.

9. Technologies Used
Python — Main programming language.
Pandas — Data manipulation and analysis.
NumPy — Numerical operations.
Scikit-learn — Machine learning model training and evaluation.
Joblib — Model serialization and loading.
10. Project Structure

A possible project structure is:

customer-churn-prediction/
│
├── dataset.csv       # Dataset containing 500 fictional customers
├── main.py           # Data preparation and model training
├── model.pkl         # Saved trained model
├── requirements.txt  # Project dependencies
└── README.md         # Project documentation


The filenames are examples and can be adapted to match the actual files in the project.

11. Installation and Usage
Prerequisites

Make sure Python 3 is installed on your machine.

Step 1: Clone the repository
git clone <repository-url>
cd customer-churn-prediction


Replace <repository-url> with the URL of your repository.

Step 2: Install dependencies
pip install pandas numpy scikit-learn joblib


Alternatively, if a requirements.txt file is available:

pip install -r requirements.txt

Step 3: Run the project

If the main Python script is named main.py, execute:

python main.py


This command should run the data preparation, training, and evaluation steps implemented in the script.

12. Limitations
The dataset contains only 500 fictional customers.
The results may not reflect real-world customer behavior.
Model performance depends on data quality and the selected features.
Accuracy alone does not fully describe the quality of churn predictions.
The model's ability to generalize should be validated using appropriate training and testing procedures.
13. Future Improvements

Possible improvements include:

Using a larger and more realistic dataset.
Testing additional machine learning algorithms.
Tuning hyperparameters to improve performance.
Evaluating the model using precision, recall, F1-score, and a confusion matrix.
Building a simple web interface to predict customer churn.
Deploying the model as an API or web application.
14. Conclusion

This project demonstrates the basic stages of a machine learning workflow, from preparing a dataset to training, evaluating, and saving a predictive model.

By using a RandomForestClassifier, we explore how customer information can be used to predict churn. The project provides a practical introduction to supervised learning, model evaluation, hyperparameter selection, and model serialization with Python.

Project type: Educational machine learning project
Task: Binary classification
Target variable: churn
Dataset size: 500 fictional customers
Model: RandomForestClassifier
Saved model: model.pkl