import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

# Fetch historical stock price data
start_date = dt.datetime(2013, 1, 1)
end_date = dt.datetime(2022, 1, 1)
df = data.get_data_stooq('AMZN', start_date, end_date)

# Sort data by date
df = df.sort_values(by=['Date']).reset_index(drop=True)

# Drop 'Date' column
df.drop(['Date'], axis=1, inplace=True)

# Plot closing prices
plt.plot(df['Close'])
plt.title('Historical Stock Prices')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.show()

# Calculate moving averages
ma100 = df['Close'].rolling(window=100).mean()
ma200 = df['Close'].rolling(window=200).mean()

# Plot moving averages
plt.plot(df['Close'], label='Close')
plt.plot(ma100, label='MA100')
plt.plot(ma200, label='MA200')
plt.title('Moving Averages')
plt.legend()
plt.show()

# Split data into training and testing sets
train_size = int(len(df) * 0.7)
train_data = df['Close'][:train_size]
test_data = df['Close'][train_size:]

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_normalized = scaler.fit_transform(np.array(train_data).reshape(-1, 1))

# Prepare data for LSTM model
X_train = []
y_train = []
for i in range(100, len(train_data_normalized)):
    X_train.append(train_data_normalized[i-100:i, 0])
    y_train.append(train_data_normalized[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120))
model.add(Dropout(0.5))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=1, batch_size=32)

# Prepare test data for prediction
inputs = df['Close'][len(df) - len(test_data) - 100:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(100,inputs.shape[0]):
    X_test.append(inputs[i-100:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

# Make predictions
predicted_stock_prices = model.predict(X_test)
predicted_stock_prices = scaler.inverse_transform(predicted_stock_prices)

# Visualize predictions
plt.plot(test_data.values, color='black', label='Actual Stock Prices')
plt.plot(predicted_stock_prices, color='green', label='Predicted Stock Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(test_data, predicted_stock_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_data, predicted_stock_prices)
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('Mean Absolute Error:', mae)
