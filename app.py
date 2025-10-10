import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

# Load model and scalers
@st.cache_resource
def load_model_components():
    """Load all model components once"""
    try:
        model = joblib.load('train/gradient_boost_model.pkl')
        feature_scaler = joblib.load('train/features_scaler.pkl')
        target_scaler = joblib.load('train/target_scaler.pkl')
        feature_columns = joblib.load('train/feature_columns.pkl')
        numerical_cols = joblib.load('train/numerical_cols.pkl')
        return model, feature_scaler, target_scaler, feature_columns, numerical_cols
    except FileNotFoundError as e:
        st.error(f"Missing file: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, feature_scaler, target_scaler, feature_columns, numerical_cols = load_model_components()

# App UI
st.title("Philippine House Price Prediction")
st.write("Enter property details to get a price prediction")



# Create input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
        bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=1)
        floor_area = st.number_input("Floor Area (sqm)", min_value=10.0, max_value=1000.0, value=50.0)
    
    with col2:
        # Only show these if your model actually uses them
        latitude = st.number_input("Latitude", value=14.5808822, format="%.7f")
        longitude = st.number_input("Longitude", value=121.05885, format="%.7f")
    
    submit = st.form_submit_button("Predict Price", type="primary")

if submit:
    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Bedrooms': [bedrooms],
            'Bath': [bath],
            'Floor_area (sqm)': [floor_area],
            'Latitude': [latitude],
            'Longitude': [longitude]
        })
        
        # Create full feature dataframe with all expected columns
        input_processed = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Fill in the numerical values
        for col in input_data.columns:
            if col in input_processed.columns:
                input_processed[col] = input_data[col].values[0]
        
        # Scale only the numerical columns
        if len(numerical_cols) > 0:
            numerical_data = input_processed[numerical_cols].copy()
            scaled_data = feature_scaler.transform(numerical_data)
            
            for i, col in enumerate(numerical_cols):
                input_processed[col] = scaled_data[0, i]
        
        # Ensure correct column order
        input_processed = input_processed[feature_columns]
        
        # Make prediction
        prediction_scaled = model.predict(input_processed)
        prediction = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
        
        # Display result
        st.success("### Prediction Complete!")
        
        col1, col2= st.columns(2)
        col1.metric("Predicted Price", f"₱{prediction:,.2f}")
        col2.metric("Floor Area", f"{floor_area} sqm")
        
        # Show input summary
        with st.expander("Input Summary"):
            summary = pd.DataFrame({
                'Feature': ['Bedrooms', 'Bathrooms', 'Floor Area', 'Latitude', 'Longitude'],
                'Value': [bedrooms, bath, f"{floor_area} sqm", latitude, longitude]
            })
            st.table(summary)
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Please check your inputs and try again")

# Footer
st.markdown("---")
st.caption("Powered by Gradient Boosting Regressor")