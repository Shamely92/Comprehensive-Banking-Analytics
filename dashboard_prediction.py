
from sklearn import preprocessing 
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from matplotlib.pyplot import figure

filename = 'final_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv("Clustered_Customer_Data.csv")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
st.title("CUSTOMER SEGMENTATION AND CREDIT SCORE PREDICTION")


with st.form("my_form"):
    Age=st.number_input(label='Age',step=0.001,format="%.6f")
    Annual_Income=st.number_input(label='Annual Income',step=0.001,format="%.6f")
    Monthly_Inhand_Salary=st.number_input(label='Monthly Inhand Salary',step=0.01,format="%.2f")
    Num_Bank_Accounts=st.number_input(label='Num Bank Accounts',step=0.01,format="%.2f")
    Num_Credit_Card=st.number_input(label='Num Credit Card',step=0.01,format="%.2f")
    Interest_Rate=st.number_input(label='Interest Rate',step=0.01,format="%.6f")
    Num_of_Loan=st.number_input(label='Num of Loan',step=0.01,format="%.6f")
    Delay_from_due_date=st.number_input(label='Delay from due date',step=0.1,format="%.6f")
    Num_of_Delayed_Payment=st.number_input(label='Num of Delayed Payment',step=0.1,format="%.6f")
    Changed_Credit_Limit=st.number_input(label='Changed Credit Limit',step=0.1,format="%.6f")
    Credit_Mix=st.number_input(label='Credit_Mix',step=1)
    Outstanding_Debt=st.number_input(label='Outstanding Debt',step=1)
    Payment_of_Min_Amount=st.number_input(label='Payment of Min Amount',step=0.1,format="%.1f")
    Total_EMI_per_month=st.number_input(label='Total EMI per month',step=0.01,format="%.6f")
    Monthly_Balance=st.number_input(label='Monthly_Balance',step=0.01,format="%.6f")
    
    data=[[Age,Annual_Income,Monthly_Inhand_Salary,Num_Bank_Accounts,Num_Credit_Card,Interest_Rate,Num_of_Loan,Delay_from_due_date,Num_of_Delayed_Payment,Changed_Credit_Limit,Credit_Mix,Outstanding_Debt,Payment_of_Min_Amount,Total_EMI_per_month,Monthly_Balance]]

    
    submitted = st.form_submit_button("Submit")

if submitted:
    clust=loaded_model.predict(data)[0]
    print('Data Belongs To ',clust)
    st.write(f"Belongs to the Cluster, {clust}")
    if clust == 'Group_A':
        st.write(f"Predicted Score:",0.99)
    elif clust == 'Group_B':
        st.write(f"Predicted Score:",0.98)
    elif clust == 'Group_C':
        st.write(f"Predicted Score:",1)
    else:
        st.write(f"Predicted Score:",0.99)




    


