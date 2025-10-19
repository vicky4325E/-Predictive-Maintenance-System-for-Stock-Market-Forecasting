# -Predictive-Maintenance-System-for-Stock-Market-Forecasting
Advanced AI-based stock market forecasting system leveraging LSTM neural networks for time series prediction and financial risk analysis
# 🧠 Predictive Maintenance System for Stock Market Forecasting

An advanced **AI-driven stock market forecasting system** designed to predict closing prices of major Indian companies (Reliance, Infosys, and TCS) using **LSTM (Long Short-Term Memory)** neural networks.  
The system combines **time-series forecasting**, **financial risk assessment**, and **sentiment analysis** to deliver accurate, real-time investment recommendations through a **React-based web interface**.

---

## 🚀 Key Features

- 📈 **LSTM-Based Stock Prediction**  
  Utilized Long Short-Term Memory networks to forecast daily closing prices for top stocks, achieving consistent accuracy across testing periods.

- 💰 **Risk Analysis & Investment Scoring**  
  Implemented standard financial formulas (Sharpe Ratio, Value at Risk, Beta Coefficient) to evaluate stock volatility and produce investment risk scores.

- 📰 **Sentiment Analysis Integration**  
  Analyzed real-time financial news headlines and market sentiment using NLP models and the `VADER` sentiment analyzer to adjust final predictions dynamically.

- 🖥️ **React + Flask Web Dashboard**  
  Developed a fully interactive web interface integrating all backend models, providing:
  - Live stock predictions  
  - Historical vs. forecasted visualization  
  - Real-time “Buy”, “Hold”, or “Sell” recommendations  

- ☁️ **Modular Architecture**  
  Backend (Flask API) and frontend (React) are decoupled for easy scaling, testing, and deployment on cloud platforms.

---

## 🧰 Tech Stack

| Component | Technology Used |
|------------|----------------|
| **Frontend** | React.js, Chart.js, TailwindCSS |
| **Backend** | Flask, REST API |
| **Machine Learning** | TensorFlow, Keras, Scikit-learn |
| **Natural Language Processing** | NLTK, VADER, TextBlob |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Plotly |
| **Deployment** | Render / Heroku (optional) |

---

## 📊 Model Overview

### 🔹 1. LSTM Model
- Designed for **time series forecasting** using previous 60 days of stock data as input.
- Data sourced via `yfinance` API.
- Optimized using **Adam optimizer** and **Mean Squared Error (MSE)** loss.
- Achieved an average **Mean Absolute Percentage Error (MAPE)** below 3%.

### 🔹 2. Risk Analysis Module
- Computes key financial metrics:
  - **Sharpe Ratio** (returns vs. volatility)
  - **Value at Risk (VaR)** (maximum expected loss)
  - **Beta Coefficient** (correlation with market)
- Outputs a normalized **Investment Score** for each stock.

### 🔹 3. Sentiment Analysis Module
- Fetches financial news using APIs (NewsAPI / FinViz).
- Applies VADER sentiment scoring to determine market tone.
- Adjusts LSTM predictions by sentiment-weighted factors.

---

## 🧮 Prediction Logic

The system aggregates three components:

