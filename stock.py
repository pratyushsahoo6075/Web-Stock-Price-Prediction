import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Set the title of the Streamlit app
st.title("Stock Price Prediction App")

# Input for the stock ticker symbol
stock_ticker = st.text_input("Enter the Stock Ticker Symbol", "GOOG")

# Define start and end dates for data retrieval
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Download stock data using yfinance
stock_data = yf.download(stock_ticker, start=start, end=end)

# Load the pre-trained Keras model
model = load_model("Latest_stock_price_model.keras")

# Display the stock data in the app
st.subheader(f"Stock Data for {stock_ticker}")
st.write(stock_data)

# Function to plot the graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange', label='Moving Average')
    plt.plot(full_data.Close, 'blue', label='Original Close Price')
    if extra_data:
        plt.plot(extra_dataset, 'green', label='Additional Data')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    return fig

# Plotting the moving averages with corrected code
moving_averages = [250, 200, 100]

for ma in moving_averages:
    ma_column = f"MA_for_{ma}_days"
    stock_data[ma_column] = stock_data['Close'].rolling(ma).mean()
    st.subheader(f"Original Close Price and MA for {ma} days")
    st.pyplot(plot_graph((15, 6), stock_data[ma_column], stock_data))

# Plot Original Close Price vs MA for 100 days and 250 days together
st.subheader("Original Close Price vs MA for 100 days and MA for 250 days")
st.pyplot(plot_graph((15, 6), stock_data["MA_for_100_days"], stock_data, 1, stock_data["MA_for_250_days"]))

# Data Preprocessing
splitting_len = int(len(stock_data) * 0.7)
x_test = pd.DataFrame(stock_data.Close[splitting_len:])

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

# Create x_data and y_data
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Make predictions
predictions = model.predict(x_data)

# Inverse the scaling of predictions and actual values
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Create a DataFrame for plotting data
index_range = stock_data.index[splitting_len+100:]
plotting_data = pd.DataFrame(
    {
        'Original Test Data': inv_y_test.flatten(),
        'Predictions': inv_predictions.flatten()
    },
    index=index_range
)

# Display the predictions vs original values
st.subheader("Original values vs Predicted values")
st.write(plotting_data)

# Plot Original Close values vs Predicted Close values
st.subheader("Original Close values vs Predicted Close values")
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([stock_data.Close[:splitting_len+100], plotting_data['Original Test Data']], axis=0), 'blue', label='Original Test Data')
plt.plot(pd.concat([pd.Series([None]*(splitting_len+100)), plotting_data['Predictions']], axis=0), 'orange', label='Predicted Test Data')
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
st.pyplot(fig)
