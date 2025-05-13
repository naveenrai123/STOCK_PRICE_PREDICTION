import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor")

# Input with validation
stock = st.text_input("Enter Stock Symbol", "AAPL").upper()

# Date range selector
end_date = pd.to_datetime('today')
start_date = end_date - pd.DateOffset(years=5)

# Download data
@st.cache_data
def load_data(symbol):
    return yf.download(symbol, start=start_date, end=end_date)

data = load_data(stock)
if data.empty:
    st.error("No data found! Check symbol.")
    st.stop()

# Feature Engineering (Simplified)
data['MA50'] = data['Close'].rolling(50).mean()
data['MA200'] = data['Close'].rolling(200).mean()
data.dropna(inplace=True)

# Prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close', 'MA50', 'MA200']])

# Create sequences
lookback = 60
X, y = [], []
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i-lookback:i])
    y.append(scaled_data[i, 0])  # Predict Close price

X, y = np.array(X), np.array(y)

# Split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Load model and predict
model = load_model("efficient_stock_model.keras")
predictions = model.predict(X_test)

# Inverse transform
predictions = scaler.inverse_transform(
    np.concatenate((predictions.reshape(-1,1), 
                   np.zeros((len(predictions), 2))), 
    axis=1))[:,0]

actual_prices = scaler.inverse_transform(
    np.concatenate((y_test.reshape(-1,1), 
                   np.zeros((len(y_test), 2))), 
    axis=1))[:,0]

# Create results DataFrame
results = pd.DataFrame({
    'Actual': actual_prices,
    'Predicted': predictions
}, index=data.index[lookback+split:])

# Visualization
st.subheader("Price Predictions")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(results.index, results['Actual'], label='Actual')
ax.plot(results.index, results['Predicted'], label='Predicted')
ax.set_title(f"{stock} Price Prediction")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Performance metrics
mae = np.mean(np.abs(results['Actual'] - results['Predicted']))
st.write(f"Mean Absolute Error: ${mae:.2f}")

# Next day prediction
last_sequence = scaled_data[-lookback:]
next_day_pred = model.predict(last_sequence.reshape(1, lookback, 3))
next_day_price = scaler.inverse_transform(
    np.concatenate((next_day_pred, np.zeros((1, 2))), axis=1))[0,0]
st.subheader(f"Next Trading Day Prediction: ${next_day_price:.2f}")import streamli
