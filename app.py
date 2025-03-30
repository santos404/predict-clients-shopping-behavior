import streamlit as st
import numpy as np
import joblib

# Load the trained K-Means model and scaler
model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit App Title and Description
st.title("Predict customer shopping behavior")
st.markdown("""
    This application predicts the cluster to which a customer belongs based on their features.
""")

# Input Fields for Prediction
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
purchase_amount = st.number_input("Purchase Amount (USD)", min_value=1.0, value=50.0, step=1.0)
previous_purchases = st.number_input("Number of Previous Purchases", min_value=0, value=10, step=1)
review_rating = st.slider("Review Rating", min_value=1.0, max_value=5.0, value=3.0, step=0.1)

# Prediction Button
if st.button("Predict Cluster"):
    # Prepare the input data
    input_data = np.array([[age, purchase_amount, previous_purchases, review_rating]])
    
    # Scale the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)
    
    # Predict the cluster using the K-Means model
    cluster = model.predict(input_data_scaled)[0]
    
    # Display the prediction result
    st.success(f"The customer belongs to Cluster: {cluster}")
    st.balloons()

st.write("Adjust the inputs and click on 'Predict Cluster' to see the result.")
