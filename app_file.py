import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import xgboost

clustering_model = 'model.sav'
loaded_model = pickle.load(open(clustering_model, 'rb'))

pca_model = 'pca_model.pkl'
loaded_pca = joblib.load(pca_model)

std_scalar = 'std_scaler.pkl'
loaded_scalar = joblib.load(std_scalar)

df = pd.read_csv("Clustered_Data.csv")

column_names = [
    'ID', 'Customer_ID', 'Month', 'Age', 'SSN', 'Annual_Income',
       'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
       'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
       'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
       'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
       'Credit_Utilization_Ratio', 'Credit_History_Age',
       'Payment_of_Min_Amount', 'Total_EMI_per_month',
       'Amount_invested_monthly', 'Monthly_Balance'
]

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(
    "<h1 style='{}'>Customer Segmentation and Prediction</h1>".format("text-align: center;"),
    unsafe_allow_html=True
)

with st.form("my_form"):
    # Create two columns
    col1, col2 = st.columns(2)

    # Input fields in the first column
    with col1:
        ID = st.number_input(label='ID', step=0.001, format="%.6f")
        Customer_ID = st.number_input(label='Customer_ID', step=0.001, format="%.6f")
        Month = st.number_input(label='Month', step=0.01, format="%.2f")
        Age = st.number_input(label='Age', step=0.01, format="%.2f")
        SSN = st.number_input(label='SSN', step=0.01, format="%.2f")
        Annual_Income = st.number_input(label='Annual_Income', step=0.01, format="%.6f")
        Monthly_Inhand_Salary = st.number_input(label='Monthly_Inhand_Salary', step=0.01, format="%.2f")
        Num_Bank_Accounts = st.number_input(label='Num_Bank_Accounts', step=0.01, format="%.2f")
        Num_Credit_Card = st.number_input(label='Num_Credit_Card', step=0.01, format="%.2f")
        Interest_Rate = st.number_input(label='Interest_Rate', step=0.01, format="%.6f")
        Num_of_Loan = st.number_input(label='Num_of_Loan', step=0.01, format="%.6f")
    # Input fields in the second column
    with col2:
        
        Delay_from_due_date = st.number_input(label='Delay_from_due_date', step=1)
        Num_of_Delayed_Payment = st.number_input(label='Num_of_Delayed_Payment', step=0.1, format="%.1f")
        Changed_Credit_Limit = st.number_input(label='Changed_Credit_Limit', step=0.01, format="%.6f")
        Num_Credit_Inquiries = st.number_input(label='Num_Credit_Inquiries', step=0.01, format="%.6f")
        Credit_Mix = st.number_input(label='Credit_Mix', step=0.01, format="%.6f")
        Outstanding_Debt = st.number_input(label='Outstanding_Debt', step=1)
        Credit_Utilization_Ratio = st.number_input(label='Credit_Utilization_Ratio', step=0.1, format="%.1f")
        Credit_History_Age = st.number_input(label='Credit_History_Age', step=0.01, format="%.6f")
        Payment_of_Min_Amount = st.number_input(label='Payment_of_Min_Amount', step=0.01, format="%.6f")
        Total_EMI_per_month = st.number_input(label='Total_EMI_per_month', step=0.01, format="%.6f")
        Amount_invested_monthly = st.number_input(label='Amount_invested_monthly', step=1)
        Monthly_Balance = st.number_input(label='Monthly_Balance', step=0.01, format="%.6f")

    input_data = [[ID,Customer_ID,Month, Age,SSN, Annual_Income,Monthly_Inhand_Salary, Num_Bank_Accounts,Num_Credit_Card,Interest_Rate, Num_of_Loan, Delay_from_due_date,Num_of_Delayed_Payment, Changed_Credit_Limit,Num_Credit_Inquiries,Credit_Mix,Outstanding_Debt,Credit_Utilization_Ratio, Credit_History_Age,Payment_of_Min_Amount, Total_EMI_per_month,Amount_invested_monthly,Monthly_Balance]]
    submitted = st.form_submit_button("Submit")

if submitted:

    scaled_data = loaded_scalar.transform(input_data)
    pca_data = loaded_pca.transform(scaled_data)
    clust=loaded_model.predict(pca_data)[0]
    print('Data Belongs to Cluster',clust)
    st.write(f"Belongs to the Cluster, {clust}")


