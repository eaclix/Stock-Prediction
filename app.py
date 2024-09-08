from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

app = Flask(__name__)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_layer = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, 50),
                torch.zeros(1, 1, 50))

    def forward(self, x):
        lstm_out, self.hidden_layer = self.lstm(x, self.hidden_layer)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

model = LSTMModel()
model.load_state_dict(torch.load('model/lstm_model.pth', map_location=torch.device('cpu')))
model.eval()

def predict_stock_price(stock_symbol):
    try:
        df = yf.download(stock_symbol, start='2013-01-01', end='2022-01-01')
        if df.empty:
            raise ValueError("No data found for the stock symbol")

        df_close = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(df_close)

        seq_length = 100
        X_test = [data_scaled[i-seq_length:i, 0] for i in range(seq_length, len(data_scaled))]
        X_test = np.array(X_test).reshape((len(X_test), seq_length, 1))

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        with torch.no_grad():
            predicted_stock_price = model(X_test_tensor)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price.numpy())

        return {
            'predicted_price': predicted_stock_price[-1][0],
            'stockPriceData': {
                'labels': list(range(len(predicted_stock_price))),
                'values': predicted_stock_price.flatten().tolist()
            },
            'historicalData': {
                'labels': list(range(len(df_close))),
                'values': df_close.flatten().tolist()
            }
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/predict')
def predict():
    stock_symbol = request.args.get('stock_symbol')
    result = predict_stock_price(stock_symbol)
    return jsonify(result)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
