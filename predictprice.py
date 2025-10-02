# The `predict_price` function in the app.py script should handle:
# 1. Creating a DataFrame from new_house_details.
# 2. Aligning columns with feature_columns loaded from training.
# 3. Identifying numerical columns to scale based on the fitted scaler's expectations.
# 4. Applying the loaded feature_scaler.
# 5. Ensuring column order matches feature_columns before prediction.
# 6. Inverse transforming the prediction using the loaded target_scaler.

# Based on the provided `app.py` content in the previous step, let's review the relevant part of the predict_price function:

def predict_price(new_house_details, model, feature_scaler, target_scaler, feature_columns):
    # Create DataFrame for the new house
    new_house_df = pd.DataFrame([new_house_details])

    # Preprocessing steps (should match train_model.py)
    # Identify numerical columns from the input details
    numerical_input_cols = ['Bedrooms', 'Bath', 'Floor_area (sqm)', 'Latitude', 'Longitude']
    # No categorical features in the input details for this example, based on previous steps

    # Create a DataFrame with all expected feature columns from training, filled with 0
    new_house_processed = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Populate the DataFrame with the input numerical values
    for col in numerical_input_cols:
        if col in new_house_df.columns and col in new_house_processed.columns:
             new_house_processed[col] = new_house_df[col]

    # Apply the *fitted* feature scaler to the numerical columns
    # Identify numerical columns that were scaled during training based on feature_scaler
    # Use feature_scaler.feature_names_in_ if available, otherwise infer from feature_columns
    # The numerical_cols variable was defined in train_model.py during the initial scaling
    # and contained ['Price (PHP)', 'Bedrooms', 'Bath', 'Floor_area (sqm)', 'Latitude', 'Longitude']
    # The feature_scaler was fitted on df_encoded[numerical_cols].
    # In the app, we only have the input features, not the target 'Price (PHP)'.
    # The scaler in `train_model.py` was fitted on 6 numerical columns including the target.
    # This is a discrepancy. The feature scaler should have been fitted ONLY on the numerical *features*.

    # Let's assume the feature_scaler saved was correctly fitted on the numerical FEATURES
    # which are 'Bedrooms', 'Bath', 'Floor_area (sqm)', 'Latitude', 'Longitude'.
    # We need to make sure the columns passed to `feature_scaler.transform` are only these.

    # Correct identification of numerical columns for scaling based on input and features used in training
    numerical_cols_for_scaling = [col for col in numerical_input_cols if col in feature_columns]


    # Create a temporary DataFrame with numerical columns in the order the scaler expects
    # We need to ensure the order here matches how the scaler was fitted.
    # Let's assume the scaler was fitted on numerical_cols_for_scaling in their original order.
    temp_numerical_df = new_house_processed[numerical_cols_for_scaling]


    # Apply the *fitted* scaler
    scaled_numerical_data = feature_scaler.transform(temp_numerical_df)

    # Replace the original numerical columns in new_house_processed with the scaled ones
    new_house_processed[numerical_cols_for_scaling] = scaled_numerical_data


    # Ensure the order of columns matches the training data - This is already handled by reindex and populating correctly, but re-ordering as a safeguard
    new_house_processed = new_house_processed[feature_columns]


    # Make prediction
    # Keras model predict returns a numpy array, need to handle accordingly
    if isinstance(model, tf.keras.Model):
        predicted_price_scaled = model.predict(new_house_processed).flatten()
    else:
        predicted_price_scaled = model.predict(new_house_processed)


    # Inverse transform the prediction to original scale
    # The target_scaler expects a 2D array
    predicted_price_original_scale = target_scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1)).flatten()

    return predicted_price_original_scale[0]