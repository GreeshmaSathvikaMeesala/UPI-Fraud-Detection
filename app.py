import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Uncomment and verify the path to your dataset
# data=pd.read_csv(r"E:\resume projects\zfraus_creditcarsd\PS_20174392719_1491204439457_log.csv")
st.sidebar.image('plots/payment_fraud.jpg', use_column_width=True)

# Load the trained models
try:
    svm = joblib.load("weights/linear_svm_model.pkl")
    logistic_regression_model = joblib.load("weights/logistic_regression_model.pkl")
    xgboost_model = joblib.load("weights/xgboost_model.pkl")
except Exception as e:
    st.error(f"Error loading models: {e}")

def intro():
    st.title('Online Payment Fraud Detection')
    st.write('Welcome to the Online Payment Fraud Detection app! This app is designed to showcase machine learning models for detecting fraud in online payment transactions.')
    st.markdown('### Dataset Overview')
    st.markdown('The dataset used in this app contains the following columns:')
    st.markdown('* **Step** : Represents a unit of time where 1 step equals 1 hour.')
    st.markdown('* **Type** : Type of online transaction.')
    st.markdown('* **Amount** : The amount of the transaction.')
    st.markdown('* **NameOrig** : Customer initiating the transaction.')
    st.markdown('* **OldbalanceOrg** : Balance before the transaction.')
    st.markdown('* **NewbalanceOrig** : Balance after the transaction.')
    st.markdown('* **NameDest** : Recipient of the transaction.')
    st.markdown('* **OldbalanceDest** : Initial balance of the recipient before the transaction.')
    st.markdown('* **NewbalanceDest** : New balance of the recipient after the transaction.')
    st.markdown('* **IsFlaggedFraud** : Indicates if the transaction has been flagged as potentially fraudulent by the system.')
    st.markdown('* **IsFraud** : Indicates whether the transaction is fraudulent (1) or not (0).')


def generate_inference_data():
    step_choices = [1, 743, 204500]
    amount_min, amount_max = 0, 92445520
    balance_min, balance_max = 0, 59585040

    inference_data = {
        'step': random.choice(step_choices),
        'amount': round(random.uniform(amount_min, amount_max), 2),
        'oldbalanceOrg': round(random.uniform(balance_min, balance_max), 2),
        'newbalanceOrig': round(random.uniform(balance_min, balance_max), 2),
        'oldbalanceDest': round(random.uniform(balance_min, balance_max), 2),
        'newbalanceDest': round(random.uniform(balance_min, balance_max), 2),
        'isFlaggedFraud': random.choice([0, 1]),
        'type_CASH_IN': 0,
        'type_CASH_OUT': 0,
        'type_DEBIT': 0,
        'type_PAYMENT': 0,
        'type_TRANSFER': 0
    }

    inference_data[random.choice(['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'])] = 1
    return pd.DataFrame([inference_data])

def predict(model, data):
    try:
        prediction = model.predict(data)[0]
        result = "Fraud" if prediction == 1 else "Legitimate"
        color = "red" if prediction == 1 else "green"
        return f"<span style='color:{color};'>{result}</span>"
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "<span style='color:gray;'>Error</span>"

def inference():
    st.title('Run Inference')
    model_selection = st.selectbox('Select Model', ['Logistic Regression','XGBoost','Support Vector Machine (SVM)'])
    if st.button('Generate and Run'):
        model = None
        if model_selection == 'Support Vector Machine (SVM)':
            model = svm
        elif model_selection == 'XGBoost':
            model = xgboost_model
        elif model_selection == 'Logistic Regression':
            model = logistic_regression_model
        if model is not None:
            inference_data = generate_inference_data()
            st.write('Generated Inference Data:')
            st.write(inference_data)
            prediction = predict(model, inference_data)
            st.markdown(f"**Prediction**: {prediction}", unsafe_allow_html=True)

def data_plots():
    plot_selection = st.selectbox('Select Plot', ['Correlation  Matrix', 'Fraud vs. Flagged Fraud', 'Fraudulent Transactions by Type', 'Transaction Types'])
    if plot_selection == 'Correlation  Matrix':
        st.image('plots/confusion_matrix.png')
    elif plot_selection == 'Fraud vs. Flagged Fraud':
        st.image('plots/fraud_flagged_counts.png')
    elif plot_selection == 'Fraudulent Transactions by Type':
        st.image('plots/fraud_transactiontype.png')
    elif plot_selection == 'Transaction Types':
        st.image('plots/transaction_types.png')

def main():
    pages = {
        "Home": intro,
        "Models": inference,
        "Data Visualisation": data_plots
    }

    st.sidebar.title('Select')
    selection = st.sidebar.radio(" ", list(pages.keys()))

    page = pages[selection]
    page()

if __name__ == "__main__":
    main()
