import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model
with open('Best_Model_1.pkl', 'rb') as file:
    model = pickle.load(file)

# Load car data for brand selection
car_data_cleaned = pd.read_csv('car_data_clean.csv')  # Adjust path to your data file

# Define categories based on your training data
fuel_types = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
seller_types = ['Dealer', 'Individual']
transmissions = ['Manual', 'Automatic']
owners = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']
brands = car_data_cleaned['brand'].unique()
model = car_data_cleaned['model']

# Streamlit app
st.title("Old Car Price Prediction")

year = st.number_input("Year", 1990, 2022)
km_driven = st.number_input("Kilometers Driven", 0, 1000000)
fuel = st.selectbox("Fuel Type", fuel_types)
seller_type = st.selectbox("Seller Type", seller_types)
transmission = st.selectbox("Transmission Type", transmissions)
owner = st.selectbox("Owner Type", owners)
brand = st.selectbox("Car Brand", brands)
model = st.selectbox("Car Model",model)

# Encoding user input
# Assuming you used integer encoding, replace this part with the appropriate encoding method used during training
fuel_type = fuel_type.index(fuel)
Type_of_Seller = seller_type.index(seller_type)
Transmit = transmission.index(transmission)
Owner_Type = owner.index(owner)
brand_name = np.where(brand == brand)[0]
model_name = np.where(model == model)[1]


input_data = np.array([
    [
        year,
        km_driven,
        fuel_type,
        Type_of_Seller,
        Transmit,
        Owner_Type,
        brand_name
    ]
])
input_data_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    st.write(f"Predicted Selling Price: {prediction[0]:.2f}")
