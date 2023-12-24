from sklearn import preprocessing 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from matplotlib.pyplot import figure
import joblib

filename = 'final_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv("Clustered_Customer_Data.csv")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load pickled models and scalers
model_credit_score = joblib.load(r'C:\Users\THEBEST\Desktop\Jupyter_workbook\sample\credit_score_model.pkl')
kmeans_model = joblib.load(r'C:\Users\THEBEST\Desktop\Jupyter_workbook\sample\kmeans_model.pkl')

# Function for Credit Score Prediction
def predict_credit_score(input_values):
    features = np.array([input_values])
    return model_credit_score.predict(features)[0]

st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
st.title("CUSTOMER SEGMENTATION AND CREDIT SCORE PREDICTION")

# Sidebar option menu
selected_option = st.sidebar.selectbox(
    "Main Menu",
    ["About Project", "Predictions"],
    format_func=lambda x: "🏠 About Project" if x == "About Project" else "🔮 Predictions"
)

# Display content based on the selected option
if selected_option == "About Project":
    st.title("About Project")
    st.markdown("""
        ## Overview:
        This prediction app utilizes machine learning models for various financial predictions. The app is designed to provide insights into customer segmentation using clustering, credit scores.

        ## Technologies Used:
        - Python
        - Streamlit
        - Scikit-learn
        - Pandas

        ## Features:
        - **Customer Segmentation:**  
          Creating a segmention of some customers from a bank based on their details and divide them into some groups through clustering algorithms and then making a prediction on those groups for new customers.    
        - **Credit Score Prediction:**
          Users can input financial data, and the app predicts their credit score using a RandomForestClassifier model.
        - **Credit Risk Analysis:**
          Predicts whether a user details leads to credit risk or not by using a RandomForestClassifier model.
               
    """)


elif selected_option == "Predictions":
    st.title("Predictions")

    # Submenu for prediction tabs
    prediction_tab = st.sidebar.selectbox(
        "Select Prediction",
        ["Predict Cluster", "Credit Score"]
    )

    # Function to get user input based on the selected tab
    def get_user_input(features):
        input_values = {}
        for feature in features:
            input_values[feature] = st.number_input(f"Enter value for {feature}:", value=0.0)
        return input_values

    # Display prediction based on the selected tab
    if prediction_tab == 'Credit Score':
        st.header('Credit Score Prediction')
        input_values_credit_score = get_user_input(['Age','Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                                                    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
                                                    'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit','Credit_Mix',
                                                    'Outstanding_Debt', 'Payment_of_Min_Amount','Total_EMI_per_month', 'Monthly_Balance'])
        
        if st.button('Predict Credit Score'):
            predicted_credit_score = predict_credit_score(list(input_values_credit_score.values()))
            st.success(f'Predicted Credit Score: {predicted_credit_score}')


    elif prediction_tab == 'Predict Cluster':          
            st.header('Clustering Prediction')
            with st.form("my_form1"):
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

        
                submitted = st.form_submit_button("Predict Cluster")

            if submitted:
                
                clust=loaded_model.predict(data)[0]
                print('Data Belongs To ',clust)
                st.write(f"Belongs to the Cluster, {clust}")

    

    



