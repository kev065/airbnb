import streamlit as st 
import joblib
import numpy as np
import pandas as pd

# Load the saved model from the pickle file
loaded_model = joblib.load('lgb_model_log.pkl')

# Load the aggregated data
capetown_aggregated_df = pd.read_csv("capetown_aggregated_df.csv")

def suggest_price(model, capetown_aggregated_df):
    # Prompt user for property details
    host_id = st.number_input("Enter your 'host ID' (e.g. 1952066, 59072 etc): ")
    host_response_rate = st.number_input("Enter host response rate (as a decimal): ", min_value=0.0, max_value=1.0, value=0.9)
    host_is_superhost = st.radio("Is the host a superhost?", ["yes", "no"])
    host_listings_count = st.number_input("Enter number of host listings: ", min_value=1, value=1)
    accommodates = st.number_input("Enter number of people the property can accommodate: ", min_value=1, value=2)
    bathrooms = st.number_input("Enter number of bathrooms: ", min_value=0.0, value=1.0)
    bedrooms = st.number_input("Enter number of bedrooms: ", min_value=0.0, value=1.0)
    beds = st.number_input("Enter number of beds: ", min_value=1, value=1)
    avg_rating = st.number_input("Enter average rating: ", min_value=0.0, max_value=5.0, value=4.5)
    number_of_reviews = st.number_input("Enter number of reviews: ", min_value=0, value=10)
    neighbourhood_cleansed = st.selectbox("Enter neighbourhood (e.g., Ward 57, 61, 64 etc): ", capetown_aggregated_df['neighbourhood_cleansed'].unique())
    property_type = st.selectbox("Enter property type (e.g., Entire home, Private room etc): ", capetown_aggregated_df['property_type'].unique())

    # Check if the neighbourhood and property type values are present in the DataFrame
    if neighbourhood_cleansed not in capetown_aggregated_df['neighbourhood_cleansed'].values:
        st.error("Invalid neighbourhood. Please enter a valid neighbourhood.")
        return
    if property_type not in capetown_aggregated_df['property_type'].values:
        st.error("Invalid property type. Please enter a valid property type.")
        return

    # Frequency encode the neighbourhood and property type
    neighbourhood_cleansed_freq = capetown_aggregated_df[capetown_aggregated_df['neighbourhood_cleansed'] == neighbourhood_cleansed]['neighbourhood_cleansed_freq'].mean()
    property_type_freq = capetown_aggregated_df[capetown_aggregated_df['property_type'] == property_type]['property_type_freq'].mean()

    # Create a DataFrame with the property details
    property_details = pd.DataFrame({
        'host_id': [host_id],
        'host_response_rate': [host_response_rate],
        'host_is_superhost': [1 if host_is_superhost.lower() == 'yes' else 0],
        'host_listings_count': [host_listings_count],
        'accommodates': [accommodates],
        'bathrooms': [bathrooms],
        'bedrooms': [bedrooms],
        'beds': [beds],
        'avg_rating': [avg_rating],
        'number_of_reviews': [number_of_reviews],
        'neighbourhood_cleansed_freq': [neighbourhood_cleansed_freq],
        'property_type_freq': [property_type_freq]
    })

    print(property_details.info())

    # Make prediction
    log_price = model.predict(property_details)
    price = np.exp(log_price[0])

    # Return suggested price
    return round(price, 2)

# Create the Streamlit app
st.title("Airbnb Price Suggestion")
st.write("Enter the details of your property to get a suggested price.")

# Call the suggest_price function and display the result
suggested_price = suggest_price(loaded_model, capetown_aggregated_df)
st.write(f"The suggested price for your property is ZAR {suggested_price}")