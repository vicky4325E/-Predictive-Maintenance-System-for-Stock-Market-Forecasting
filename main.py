# import numpy as np
# import pandas as pd
# import yfinance as yf
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, r2_score
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# import matplotlib.pyplot as plt

# # Define the function for loading and preparing data
# def load_data(symbol, start_date="2010-01-01", end_date="2025-1-30"):
#     data = yf.download(symbol, start=start_date, end=end_date)
#     data = data[['Close']]
#     return data

# # Define a function for preprocessing
# def preprocess_data(data):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data)
#     return scaled_data, scaler

# # Prepare data in sequences
# def create_sequences(data, time_steps=60):
#     X, y = [], []
#     for i in range(time_steps, len(data)):
#         X.append(data[i-time_steps:i, 0])
#         y.append(data[i, 0])
#     return np.array(X), np.array(y)

# # Load data for top 3 NSE companies (replace 'RELIANCE.NS' with actual top 3 symbols)
# symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
# datasets = {symbol: load_data(symbol) for symbol in symbols}

# # Preprocess each dataset
# preprocessed_data = {}
# for symbol, data in datasets.items():
#     scaled_data, scaler = preprocess_data(data)
#     X, y = create_sequences(scaled_data)
#     preprocessed_data[symbol] = (X, y, scaler)

# # Define and compile the model
# def create_model(input_shape):
#     model = Sequential([
#         LSTM(100, return_sequences=True, input_shape=(input_shape[1], 1)),
#         Dropout(0.2),
#         LSTM(50, return_sequences=False),
#         Dropout(0.2),
#         Dense(25),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# # Train, evaluate, and predict for each company
# results = {}
# time_steps = 60
# for symbol, (X, y, scaler) in preprocessed_data.items():
#     # Split data into training and testing sets
#     split = int(0.8 * len(X))
#     X_train, X_test = X[:split], X[split:]
#     y_train, y_test = y[:split], y[split:]

#     # Reshape for LSTM input
#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#     # Create and train the model
#     model = create_model(X_train.shape)
#     early_stop = EarlyStopping(monitor='loss', patience=10)
#     model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=[early_stop])

#     # Predictions
#     predictions = model.predict(X_test)
#     predictions = scaler.inverse_transform(predictions)
#     y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

#     # Calculate errors
#     mse = mean_squared_error(y_test_rescaled, predictions)
#     r2 = r2_score(y_test_rescaled, predictions)

#     # Predict next day
#     last_sequence = X_test[-1]
#     next_day_pred = model.predict(np.expand_dims(last_sequence, axis=0))
#     next_day_price = scaler.inverse_transform(next_day_pred)[0][0]

#     # Store results and prepare for visualization
#     results[symbol] = {
#         "MSE": mse,
#         "R2": r2,
#         "Next Day Price": next_day_price,
#         "Predictions": predictions,
#         "Actual": y_test_rescaled
#     }

#     # Error Visualization
#     plt.figure(figsize=(14, 5))
#     plt.plot(y_test_rescaled, color='blue', label='Actual Prices')
#     plt.plot(predictions, color='red', label='Predicted Prices')
#     plt.title(f'{symbol} Price Prediction')
#     plt.xlabel('Time')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.show()

# # Display results
# for symbol, metrics in results.items():
#     print(f"Results for {symbol}:")
#     print(f"MSE: {metrics['MSE']}")
#     print(f"R2 Score: {metrics['R2']}")
#     print(f"Predicted Next Day Price: {metrics['Next Day Price']}\n")
from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
import yfinance as yf
from pydantic import BaseModel

# Load Models
models = {
    "reliance": tf.keras.models.load_model("backend/models/RELIANCE.NS.h5"),
    "infosys": tf.keras.models.load_model("backend/models/INFY.NS.h5"),
    "tcs": tf.keras.models.load_model("backend/models/TCS.NS.h5"),
}

risk_model = tf.keras.models.load_model("backend/models/riskmanagement.h5")
sentiment_model = tf.keras.models.load_model("backend/models/sentiment_analysis.h5")

app = FastAPI()

class StockInput(BaseModel):
    features: list[float]  # Assuming input is a list of float values

# Mapping company names to their Yahoo Finance ticker symbols
ticker_mapping = {
    "reliance": "RELIANCE.NS",
    "infosys": "INFY.NS",
    "tcs": "TCS.NS",
}

def get_real_time_price(ticker):
    """Fetches the real-time stock price from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period="1d")["Close"].iloc[-1]  # Get last closing price
    except Exception as e:
        return None  # Handle errors if Yahoo Finance API fails

@app.get("/")
def home():
    return {"message": "Alpha Insights API is running"}

@app.post("/predict/{company}")
def predict_stock(company: str, data: StockInput):
    if company not in models:
        raise HTTPException(status_code=404, detail="Company model not found")

    model = models[company]
    input_data = np.array(data.features).reshape(1, -1)

    # Predict stock price
    predicted_price = model.predict(input_data)[0][0]

    # Get real-time actual stock price
    actual_price = get_real_time_price(ticker_mapping[company])
    if actual_price is None:
        return {"error": "Failed to fetch real-time price"}

    # Predict risk score
    risk_score = risk_model.predict(input_data)[0][0]
    risk_level = "Low" if risk_score < 0.4 else "Medium" if risk_score < 0.7 else "High"

    # Predict sentiment
    sentiment_score = sentiment_model.predict(input_data)[0][0]
    sentiment = "Positive" if sentiment_score > 0.5 else "Negative"

    return {
        "company": company,
        "actual_price": round(actual_price, 2),
        "predicted_price": round(predicted_price, 2),
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "sentiment": sentiment
    }
