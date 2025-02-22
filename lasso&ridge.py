import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://ted678:Orange@cluster0.yqm4d.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"


# Load pre-trained models
def load_models():
    with open("lasso3.joblib", "rb") as file:
        lasso_model, scaler = joblib.load(file)
    with open("ridge5.joblib", "rb") as file:
        ridge_model, _ = joblib.load(file) 
    return lasso_model, ridge_model, scaler

# Preprocessing function
def preprocess_input(data, scaler):
    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df)
    return df_scaled

# Prediction function
def predict(data):
    lasso_model, ridge_model, scaler = load_models()
    processed_data = preprocess_input(data, scaler)
    
    # Predictions
    lasso_pred = lasso_model.predict(processed_data)[0]
    ridge_pred = ridge_model.predict(processed_data)[0]
    
    return round(float(lasso_pred), 2), round(float(ridge_pred), 2)

# Streamlit UI
def main():
    st.title("Diabetes Prediction (Lasso & Ridge)")
    st.write("Enter your data and predict diabetes")


    age = st.slider("Age (normalized)", -0.2, 0.2, 0.0, 0.01)
    sex = st.slider("Sex (normalized)", -0.1, 0.1, 0.0, 0.01)
    bmi = st.slider("BMI (Body Mass Index)", -0.1, 0.2, 0.0, 0.01)
    bp = st.slider("Blood Pressure", -0.1, 0.2, 0.0, 0.01)
    s1 = st.slider("Serum Level 1", -0.2, 0.2, 0.0, 0.01)
    s2 = st.slider("Serum Level 2", -0.2, 0.2, 0.0, 0.01)
    s3 = st.slider("Serum Level 3", -0.2, 0.2, 0.0, 0.01)
    s4 = st.slider("Serum Level 4", -0.2, 0.2, 0.0, 0.01)
    s5 = st.slider("Serum Level 5", -0.2, 0.2, 0.0, 0.01)
    s6 = st.slider("Serum Level 6", -0.2, 0.2, 0.0, 0.01)

    if st.button("Predict Diabetes: "):
        user_data = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "bp": bp,
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "s4": s4,
            "s5": s5,
            "s6": s6
        }
        
        # Get predictions
        lasso_pred, ridge_pred = predict(user_data)
        user_data['lasso_prediction'] = lasso_pred
        user_data['ridge_prediction'] = ridge_pred
        
        st.success(f"Lasso Prediction: {lasso_pred}")
        st.success(f"Ridge Prediction: {ridge_pred}")
        

if __name__ == "__main__":
    main()
