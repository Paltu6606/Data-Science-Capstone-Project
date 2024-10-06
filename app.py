import streamlit as st
import pandas as pd
import pickle  # Use pickle instead of joblib

# Load your trained model
with open('Best_Model_1.pkl', 'rb') as file:
    model = pickle.load(file)



# Load your trained model
# model = joblib.load('Best_Model_1.pkl')  # Change this to your model's filename

# Title of the app
st.title("Car Price Prediction App")

# Sidebar for user input
st.sidebar.header("User Input Features")

# Function to get user input
def get_user_input():
    brand = st.sidebar.selectbox("Brand", options=["Maruti","Hyundai","Hyundai","Mahindra", "Mahindra", "Tata", "Ford", "Honda" ,"Toyota", "Chevrolet",
                                                   "Renault","Volkswagen", "Nissan", "Skoda", "Fiat", "Audi", "Datsun", "BMW", "Mercedes-Benz", "Jaguar",
                                                   "Mitsubishi", "Land", "Volvo", "Jeep", "Ambassador" , "MG", "OpelCorsa", "Daewoo", "Force", "Isuzu",
                                                   "Kia"])
    year = st.sidebar.slider("Year", min_value=1992, max_value=2020, value=1992)
    mileage = st.sidebar.slider("KM Driven (in km)", min_value=0, max_value=806599, value=0)
    fuel_type = st.sidebar.selectbox("Fuel Type", options=["Petrol", "Diesel", "Electric", "LPG", "CNG"])
    transmission = st.sidebar.selectbox("Transmission", options=["Manual", "Automatic"])
    return pd.DataFrame({
        'Brand': [brand],
        'Year': [year],
        'Mileage': [mileage],
        'Fuel_Type': [fuel_type],
        'Transmission': [transmission]
    })

# Get user input
input_df = get_user_input()

# Show user input
st.write("User Input:")
st.write(input_df)

# Make predictions
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.write(f"Predicted Price: ${prediction[0]:,.2f}")

# Optionally, display a sample of the dataset
if st.checkbox("Show Sample Data"):
    sample_data = pd.read_csv("car_details.csv")  # Change to your actual dataset file
    st.write(sample_data.sample(10))

# Add any other visualizations you want (e.g., feature importance, etc.)

