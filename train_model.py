from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import torch
import torch.nn as nn

# Define the LSTM model in PyTorch
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# Flask app setup
app = Flask(__name__)

# Load the PyTorch model
model = LSTM()
model.load_state_dict(torch.load('model/lstm_model.pth'))
model.eval()

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
    X_test = torch.from_numpy(X_test).float()

    # Predict
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        predicted_stock_price = model(X_test[-1])

    predicted_stock_price = scaler.inverse_transform(predicted_stock_price.detach().numpy().reshape(-1, 1))

    return predicted_stock_price[0][0]

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
