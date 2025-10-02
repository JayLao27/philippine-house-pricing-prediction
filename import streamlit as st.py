import streamlit as st
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

@st.cache_data
def load_and_train_model():
    df = pd.read_csv('train.csv')

    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    columns_to_drop = ['Location_Last_Part', 'Description', 'Link', 'Price_Category']
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    df.drop_duplicates(inplace=True)

    categorical_cols = df.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    target = df_encoded['Price (PHP)']
    features_encoded = df_encoded.drop('Price (PHP)', axis=1)

    numerical_features_cols = features_encoded.select_dtypes(include=np.number).columns

    scaler = StandardScaler()
    features_encoded[numerical_features_cols] = scaler.fit_transform(features_encoded[numerical_features_cols])

    features_train, features_test, target_train, target_test = train_test_split(
        features_encoded, target, test_size=0.3, random_state=42
    )

    gradient_boost_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gradient_boost_model.fit(features_train, target_train)

    return gradient_boost_model, scaler, features_train.columns.tolist(), numerical_features_cols.tolist(), categorical_cols.tolist()

def predict_price(new_house_details, model, scaler, feature_columns, numerical_cols, categorical_cols):
    new_house_df = pd.DataFrame([new_house_details])
    
    cols_to_encode_in_new_house = [col for col in categorical_cols if col in new_house_df.columns]
    
    new_house_encoded = pd.get_dummies(new_house_df, columns=cols_to_encode_in_new_house, drop_first=True)
    
    new_house_processed = new_house_encoded.reindex(columns=feature_columns, fill_value=0)
    
    numerical_cols_to_scale = [col for col in numerical_cols if col in new_house_processed.columns]
    
    new_house_processed[numerical_cols_to_scale] = scaler.transform(new_house_processed[numerical_cols_to_scale])
    
    new_house_processed = new_house_processed[feature_columns]
    
    predicted_price = model.predict(new_house_processed)
    
    return predicted_price[0]

try:
    if all(os.path.exists(f) for f in ['gradient_boost_model.pkl', 'scaler.pkl', 'feature_columns.pkl', 'numerical_cols.pkl', 'categorical_cols.pkl']):
        gradient_boost_model = joblib.load('gradient_boost_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        numerical_cols = joblib.load('numerical_cols.pkl')
        categorical_cols = joblib.load('categorical_cols.pkl')
        st.success("Loaded pre-trained model!")
    else:
        st.info("Training new model...")
        gradient_boost_model, scaler, feature_columns, numerical_cols, categorical_cols = load_and_train_model()
        
        joblib.dump(gradient_boost_model, 'gradient_boost_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(feature_columns, 'feature_columns.pkl')
        joblib.dump(numerical_cols, 'numerical_cols.pkl')
        joblib.dump(categorical_cols, 'categorical_cols.pkl')
        st.success("Model trained and saved!")

except Exception as e:
    st.error(f"Error loading/training model: {str(e)}")
    st.stop()

st.title("Philippine House Price Prediction")
st.write("Enter the details of the house to get a price prediction using Gradient Boosting Regressor.")

bedrooms = st.number_input("Number of Bedrooms", min_value=1, value=2)
bath = st.number_input("Number of Bathrooms", min_value=1, value=2)
floor_area = st.number_input("Floor Area (sqm)", min_value=10.0, value=106.0)
latitude = st.number_input("Latitude", value=14.575822)
longitude = st.number_input("Longitude", value=121.064324)

new_house_details = {
    'Bedrooms': bedrooms,
    'Bath': bath,
    'Floor_area (sqm)': floor_area,
    'Latitude': latitude,
    'Longitude': longitude
}

if st.button("Predict Price"):
    try:
        predicted_price = predict_price(new_house_details, gradient_boost_model, scaler, feature_columns, numerical_cols, categorical_cols)
        st.success(f"Predicted House Price: ₱{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")