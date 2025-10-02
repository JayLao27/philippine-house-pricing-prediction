# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os
import streamlit as st
@st.cache_data
def load_and_train_model():
    """Load data, train model, and return trained components"""
    
    # Load the data
    df = pd.read_csv('train.csv')

    # Handle missing values
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Remove redundant columns
    columns_to_drop = ['Location_Last_Part', 'Description', 'Link', 'Price_Category']
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Define features and target
    target = df['Price (PHP)']
    features = df.drop('Price (PHP)', axis=1)

    # Identify categorical and numerical columns
    categorical_cols = features.select_dtypes(include='object').columns
    numerical_cols = features.select_dtypes(include=np.number).columns

    # Apply one-hot encoding
    features_encoded = pd.get_dummies(features, columns=categorical_cols, drop_first=True)

    # Apply standard scaling
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(features_encoded[numerical_cols])
    features_scaled = pd.DataFrame(features_scaled, columns=numerical_cols, index=features_encoded.index)

    # Combine scaled numerical and encoded categorical features
    features_processed = pd.concat([features_scaled, features_encoded.drop(numerical_cols, axis=1)], axis=1)

    # Scale the target variable
    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1)).flatten()

    # Split data into training and testing sets
    features_train, features_test, target_train_scaled, target_test_scaled = train_test_split(
        features_processed, target_scaled, test_size=0.3, random_state=42
    )

    # Train Gradient Boosting Regressor
    gradient_boost_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gradient_boost_model.fit(features_train, target_train_scaled)

    return gradient_boost_model, feature_scaler, target_scaler, features_processed.columns.tolist(), numerical_cols.tolist()

def predict_price(new_house_details, model, feature_scaler, target_scaler, feature_columns, numerical_cols):
    """Make prediction for new house data"""
    
    # Create DataFrame for the new house
    new_house_df = pd.DataFrame([new_house_details])
    
    # Create a DataFrame with all expected feature columns, filled with 0
    new_house_processed = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Input numerical columns
    numerical_input_cols = ['Bedrooms', 'Bath', 'Floor_area (sqm)', 'Latitude', 'Longitude']
    
    # Populate the DataFrame with the input numerical values
    for col in numerical_input_cols:
        if col in new_house_df.columns and col in new_house_processed.columns:
            new_house_processed[col] = new_house_df[col].values[0]
    
    # Apply scaling only to the numerical columns that were scaled during training
    temp_numerical_df = new_house_processed[numerical_cols].copy()
    
    # Scale the numerical data
    scaled_numerical_data = feature_scaler.transform(temp_numerical_df)
    
    # Replace the original numerical columns with the scaled ones
    for i, col in enumerate(numerical_cols):
        new_house_processed[col] = scaled_numerical_data[0, i]
    
    # Ensure the order of columns matches the training data
    new_house_processed = new_house_processed[feature_columns]
    
    # Make prediction
    predicted_price_scaled = model.predict(new_house_processed)
    
    # Inverse transform the prediction to original scale
    predicted_price_original_scale = target_scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1)).flatten()
    
    return predicted_price_original_scale[0]

# Load or train the model
try:
    # Try to load existing model files
    if all(os.path.exists(f) for f in ['gradient_boost_model.pkl', 'feature_scaler.pkl', 'target_scaler.pkl', 'feature_columns.pkl', 'numerical_cols.pkl']):
        gradient_boost_model = joblib.load('gradient_boost_model.pkl')
        feature_scaler = joblib.load('feature_scaler.pkl')
        target_scaler = joblib.load('target_scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        numerical_cols = joblib.load('numerical_cols.pkl')
        st.success("Loaded pre-trained model!")
    else:
        # Train new model if files don't exist
        st.info("Training new model...")
        gradient_boost_model, feature_scaler, target_scaler, feature_columns, numerical_cols = load_and_train_model()
        
        # Save the trained components
        joblib.dump(gradient_boost_model, 'gradient_boost_model.pkl')
        joblib.dump(feature_scaler, 'feature_scaler.pkl')
        joblib.dump(target_scaler, 'target_scaler.pkl')
        joblib.dump(feature_columns, 'feature_columns.pkl')
        joblib.dump(numerical_cols, 'numerical_cols.pkl')
        st.success("Model trained and saved!")

except Exception as e:
    st.error(f"Error loading/training model: {str(e)}")
    st.stop()

# Streamlit App
st.title("Philippine House Price Prediction")
st.write("Enter the details of the house to get a price prediction using Gradient Boosting Regressor.")

# Input fields for house details
bedrooms = st.number_input("Number of Bedrooms", min_value=1, value=2)
bath = st.number_input("Number of Bathrooms", min_value=1, value=1)
floor_area = st.number_input("Floor Area (sqm)", min_value=10.0, value=126.0)
latitude = st.number_input("Latitude", value=14.5808822)
longitude = st.number_input("Longitude", value=121.05885)

new_house_details = {
    'Bedrooms': bedrooms,
    'Bath': bath,
    'Floor_area (sqm)': floor_area,
    'Latitude': latitude,
    'Longitude': longitude
}

if st.button("Predict Price"):
    try:
        predicted_price = predict_price(new_house_details, gradient_boost_model, feature_scaler, target_scaler, feature_columns, numerical_cols)
        st.success(f"Predicted House Price: ₱{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
