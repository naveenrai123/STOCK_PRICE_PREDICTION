import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor")

# Fix 1: Add proper model loading with error handling
try:
    # Fix 2: Use raw string for Windows path
    model_path = r"C:\Users\vishe\Downloads\Latest_stock_price_model.keras"
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Input with validation
stock = st.text_input("Enter Stock Symbol", "AAPL").upper()

# Date range selector
end_date = pd.to_datetime('today')
start_date = end_date - pd.DateOffset(years=5)

# Download data
@st.cache_data
def load_data(symbol):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

data = load_data(stock)
if data.empty:
    st.error("No data found! Check symbol.")
    st.stop()

# Feature Engineering (Simplified)
data['MA50'] = data['Close'].rolling(50).mean()
data['MA200'] = data['Close'].rolling(200).mean()
data.dropna(inplace=True)

# Prepare data
# Fix 3: Only scale the Close price for simplicity
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(data[['Close']])

# Create sequences
lookback = 60
X, y = [], []
for i in range(lookback, len(scaled_close)):
    X.append(scaled_close[i-lookback:i])
    y.append(scaled_close[i, 0])  # Predict Close price

X, y = np.array(X), np.array(y)

# Split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Fix 4: Make predictions
try:
    predictions = model.predict(X_test)
except Exception as e:
    st.error(f"Prediction failed: {str(e)}")
    st.stop()

# Inverse transform predictions
# Fix 5: Simplified inverse transform since we only scaled Close price
predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Create results DataFrame
# Fix 6: Correct index calculation
results = pd.DataFrame({
    'Actual': actual_prices,
    'Predicted': predictions
}, index=data.index[lookback+split:len(data)-len(y_test)+len(predictions)])

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
# Fix 7: Use proper sequence for prediction
try:
    last_sequence = scaled_close[-lookback:]
    next_day_pred = model.predict(last_sequence.reshape(1, lookback, 1))
    next_day_price = scaler.inverse_transform(next_day_pred)[0][0]
    st.subheader(f"Next Trading Day Prediction: ${next_day_price:.2f}")
except Exception as e:
    st.error(f"Next day prediction failed: {str(e)}")
