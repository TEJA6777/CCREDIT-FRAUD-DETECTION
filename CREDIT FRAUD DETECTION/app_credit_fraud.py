import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Function to load pickled objects
def load_pickled_objects():
    label_encoder = pickle.load(open('label.pkl','rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl','rb'))
    return label_encoder, model, scaler

# Function for preprocessing input data
def preprocess_input(data, label_encoder, scaler):
    # Perform label encoding for categorical columns
    categorical_cols = ['merchant', 'category', 'first', 'last', 'street', 'city', 'state', 'zip', 'F', 'M', 'job']
    for col in categorical_cols:
        data[col] = label_encoder.transform(data[col])

    # Perform scaling for numerical columns
    numerical_cols = ['cc_num', 'amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'age',
                      'trans_year', 'trans_month', 'trans_day', 'trans_hour', 'trans_minute']
    data[numerical_cols] = scaler.transform(data[numerical_cols])

    return data

# Load pickled objects
label_encoder, model, scaler = load_pickled_objects()

# Streamlit interface
st.title("CREDIT FRAUD DETECTION")

# Input fields
cc_num = st.text_input("Enter the cc_num")
merchant = st.text_input("Enter the merchant")
category = st.text_input("Enter the category")
amt = st.text_input("Enter the amount")
first = st.text_input("Enter the first Name")
last = st.text_input("Enter the last Name")
street = st.text_input("Enter the street")
city = st.text_input("Enter the city")
state = st.text_input("Enter the state")
zip_code = st.text_input("Enter the zip")
lat = st.text_input("Enter the latitude")
long = st.text_input("Enter the longitude")
city_pop = st.text_input("Enter the city population")
job = st.text_input("Enter the job")
unix_time = st.text_input("Enter the unix_time")
merch_lat = st.text_input("Enter the merch_lat")
merch_long = st.text_input("Enter the merch_long")
age = st.text_input("Enter the age")
trans_year = st.text_input("Enter the transaction year")
trans_month = st.text_input("Enter the transaction month")
trans_day = st.text_input("Enter the transaction day")
trans_hour = st.text_input("Enter the transaction hour")
trans_minute = st.text_input("Enter the transaction minute")
F = st.text_input("Enter 1 for Female or 0 for Male")
M = st.text_input("Enter 1 for Male or 0 for Female")


# Prediction button
if st.button("Predict"):
    # Create a dictionary with user inputs
    input_data = {
        'cc_num': cc_num, 'merchant': merchant, 'category': category, 'amt': amt,
        'first': first, 'last': last, 'street': street, 'city': city, 'state': state,
        'zip': zip_code, 'lat': lat, 'long': long, 'city_pop': city_pop, 'job': job,
        'unix_time': unix_time, 'merch_lat': merch_lat, 'merch_long': merch_long,
        'age': age, 'trans_year': trans_year, 'trans_month': trans_month,
        'trans_day': trans_day, 'trans_hour': trans_hour, 'trans_minute': trans_minute,
        'F': F, 'M': M
    }
    print(input_data)
    # Convert input_data to a DataFrame
    input_df = pd.DataFrame([input_data])
    print(input_data)
    # Preprocess the input data
    input_df_processed = preprocess_input(input_df, label_encoder, scaler)
    print(input_data)
    # Predict using the model
    prediction = model.predict(input_df_processed)

    # Display prediction result
    st.write("Prediction:", prediction)
