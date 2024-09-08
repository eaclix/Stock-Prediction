from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import yfinance as yf

app = Flask(__name__)

# Load the trained LSTM model
model = load_model('model/lstm_model.h5')

# Prepare function to make predictions
def predict_stock_price(stock_symbol):
    # Fetch recent stock data
    df = yf.download(stock_symbol, start='2013-01-01', end='2022-01-01')
    df = df['Close'].values.reshape(-1, 1)

    # Preprocess data (same as during training)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df)
    
    # Use the same sequence length as during training
    seq_length = 100
    X_test = []
    for i in range(seq_length, len(data_scaled)):
        X_test.append(data_scaled[i-seq_length:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predict
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    return predicted_stock_price[-1][0]

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']
        predicted_price = predict_stock_price(stock_symbol)
        return render_template('index.html', predicted_price=predicted_price, stock_symbol=stock_symbol)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
