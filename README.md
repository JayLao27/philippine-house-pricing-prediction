# 🏠 Philippine House Price Prediction

A machine learning application that predicts house prices in the Philippines based on property features like bedrooms, bathrooms, floor area, and location coordinates.

## 📋 Overview

This project uses a **Gradient Boosting Regressor** model to predict house prices across major Philippine cities including Manila, Davao, Cebu, and surrounding areas. The model was trained on real estate data and achieves an R² score of **0.935**, indicating high prediction accuracy.

## 🌐 Live Demo

Try the application here: **[Philippine House Price Predictor](https://jaylao27-philippine-house-pricing-prediction-app-m49o5v.streamlit.app/)**

## ✨ Features

- **Real-time Price Prediction**: Get instant price estimates based on property characteristics
- **Interactive Web Interface**: User-friendly Streamlit application
- **Multiple Model Training**: Comparison of various ML models (Random Forest, XGBoost, Neural Network, Gradient Boosting)
- **Data Preprocessing**: Automated feature scaling and encoding
- **Model Performance Metrics**: R² scores and MSE evaluation

## 🚀 How to Use the Application

### Online (Recommended)

1. Visit the live application: [https://jaylao27-philippine-house-pricing-prediction-app-m49o5v.streamlit.app/](https://jaylao27-philippine-house-pricing-prediction-app-m49o5v.streamlit.app/)
2. Enter property details:
   - **Bedrooms**: Number of bedrooms (1-10)
   - **Bathrooms**: Number of bathrooms (1-10)
   - **Floor Area**: Size in square meters (10-1000 sqm)
   - **Latitude**: Geographic latitude coordinate
   - **Longitude**: Geographic longitude coordinate
3. Click **"Predict Price"** button
4. View the predicted price in Philippine Peso (₱)

### Local Installation

#### Prerequisites

- Python 3.8 or higher
- pip package manager

#### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/jaylao27/philippine-house-pricing-prediction.git
cd philippine-house-pricing-prediction
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit application**
```bash
streamlit run app.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
philippine-house-pricing-prediction/
│
├── app.py                          # Streamlit web application
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── xgboost_model.json             # Trained XGBoost model
│
├── data/                          # Dataset files
│   ├── final_cleaned_house_data.csv
│   ├── PH_houses_v2.csv
│   ├── test.csv
│   └── train.csv
│
├── notebook/                      # Jupyter notebooks
│   └── train_model.ipynb         # Model training and evaluation
│
└── train/                        # Trained models and scalers
    ├── gradient_boost_model.pkl  # Main prediction model
    ├── features_scaler.pkl       # Feature scaler
    ├── target_scaler.pkl         # Target scaler
    ├── feature_columns.pkl       # Feature column names
    ├── numerical_cols.pkl        # Numerical column names
    └── neural_network_model.h5   # Neural network model
```

## 🔧 Technical Details

### Models Trained

The project evaluates multiple machine learning models:

| Model | R² Score | MSE |
|-------|----------|-----|
| **Random Forest** | 0.9531 | 4.66e+12 |
| **XGBoost** | 0.9531 | 4.66e+12 |
| **Gradient Boosting** ⭐ | 0.9355 | 6.40e+12 |
| **Linear Regression** | 0.8492 | 1.50e+13 |
| **Neural Network** | -2.222 | 3.20e+14 |

**Selected Model**: Gradient Boosting Regressor (deployed in production)

### Data Preprocessing

1. **Missing Value Handling**: Imputation with mean (numerical) and mode (categorical)
2. **Feature Engineering**: 
   - One-hot encoding for categorical variables
   - Standard scaling for numerical features
3. **Feature Selection**: 82 total features after encoding
4. **Train-Test Split**: 70% training, 30% testing

### Model Features

The model uses the following input features:
- `Bedrooms`: Number of bedrooms
- `Bath`: Number of bathrooms
- `Floor_area (sqm)`: Property size in square meters
- `Latitude`: Geographic latitude
- `Longitude`: Geographic longitude
- Location-based encoded features (automatically handled)

## 📊 Model Performance

### Gradient Boosting Regressor Metrics
- **Training R²**: 0.9911
- **Test R²**: 0.9355
- **Overfitting Difference**: 0.0556 (minimal overfitting)

### Model Comparison Summary
The Gradient Boosting model was selected for deployment based on:
- High R² score indicating good predictive power
- Low overfitting compared to Random Forest
- Better generalization than Neural Network
- Consistent performance across training and test sets

## 🛠️ Development

### Training New Models

1. Open the Jupyter notebook:
```bash
jupyter notebook notebook/train_model.ipynb
```

2. Run all cells to:
   - Load and preprocess data
   - Train multiple models
   - Compare performance
   - Save trained models to `train/` directory

### Adding New Features

To add new features to the model:

1. Update data preprocessing in [`notebook/train_model.ipynb`](notebook/train_model.ipynb)
2. Retrain the model with new features
3. Update [`app.py`](app.py) to accept new input features
4. Save updated scalers and feature columns

## 📦 Dependencies

Main packages used:
- `streamlit`: Web application framework
- `scikit-learn`: Machine learning algorithms
- `xgboost`: Gradient boosting implementation
- `tensorflow/keras`: Neural network framework
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `joblib`: Model serialization

See [`requirements.txt`](requirements.txt) for complete list.

## 🎯 Use Cases

- **Real Estate Agents**: Quick property valuation
- **Home Buyers**: Price estimation before negotiation
- **Property Investors**: Market analysis and investment decisions
- **Developers**: Pricing strategy for new projects

## 📈 Future Improvements

 Add more location-specific features (neighborhood, amenities)
 Include property age and condition
 Implement time-series analysis for price trends
 Add confidence intervals for predictions
 Expand dataset to more Philippine cities
 Mobile application development

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👨‍💻 Author

**Jay Lao**
- GitHub: [@jaylao27](https://github.com/jaylao27)
- Application: [Live Demo](https://jaylao27-philippine-house-pricing-prediction-app-m49o5v.streamlit.app/)

## 🙏 Acknowledgments

- Dataset sourced from Philippine real estate listings
- Built with Streamlit, scikit-learn, and XGBoost
- Inspired by real estate market analysis needs in the Philippines

---

**Note**: Predictions are estimates based on historical data and should be used as a reference only. Actual market prices may vary based on additional factors not captured in the model.


