<<<<<<< HEAD
<h1>UPI Fraud Detection</h1>
This repository contains a fraud detection model for online transactions using machine learning algorithms.

<h3>Dataset</h3>
The dataset used for training and testing the model contains online transaction data. It includes the following columns:

step: Represents a unit of time where 1 step equals 1 hour.
type: Type of online transaction.
amount: The amount of the transaction.
nameOrig: Customer starting the transaction.
oldbalanceOrg: Balance before the transaction.
newbalanceOrig: Balance after the transaction.
nameDest: Recipient of the transaction.
oldbalanceDest: Initial balance of recipient before the transaction.
newbalanceDest: The new balance of recipient after the transaction.
isFraud: Indicates whether the transaction is fraudulent (1) or not (0).

<h3>Model</h3>
The model uses several machine learning algorithms, including Support Vector Machine (SVM), Logistic Regression, and XGBoost, to classify transactions as fraudulent or non-fraudulent. To handle the imbalanced nature of the dataset, Synthetic Minority Over-sampling Technique (SMOTE) is used.

The models achieved high accuracy, precision, and recall scores on the test set.

<h3>Deployment</h3>
The fraud detection model is deployed using Streamlit.

Web Application Link: https://online-payment-fraud-detection-tcgl9hpjt6zqbd6mwwmedn.streamlit.app/

<h3>Usage</h3>
To use the app, select a model (SVM, Logistic Regression, or XGBoost) and click "Generate and Run" to generate inference data and predict whether the transaction is fraudulent or legitimate.

<h3>Data Visualisation</h3>
The app includes data visualisation features to explore the dataset and model performance, including a correlation matrix, fraud vs. flagged fraud plot, fraudulent transactions by type, and transaction types plot.
=======
# UPI-Fraud-Detection
>>>>>>> 9104f9fd9b1cd53ca237c0c79e3e9f9e1d77af69
