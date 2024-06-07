# Stock Trend Prediction using LSTM

This project implements stock trend prediction using Long Short-Term Memory (LSTM) networks. It utilizes historical stock price data to train an LSTM model, which can then predict future stock prices based on the learned patterns.

## Overview

The project follows these main steps:

1. **Data Collection**: Historical stock price data for a chosen stock (e.g., AMZN) is fetched using the pandas-datareader library from sources like Yahoo Finance.

2. **Data Preprocessing**: The data is cleaned, sorted, and normalized. Moving averages (MA) are calculated to extract relevant features.

3. **Dataset Preparation**: The dataset is split into training and testing sets, and sequences of input data along with their corresponding target data are prepared for training the LSTM model.

4. **Model Building and Training**: An LSTM model with multiple layers and dropout regularization is constructed. The model is compiled and trained using the training dataset.

5. **Prediction**: After training, the model is used to predict future stock prices based on the testing dataset.

6. **Evaluation**: The performance of the model is evaluated using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/stock_prediction_lstm.git

2. Navigate to the project directory
    ```bash
    cd stock_prediction_lstm

3 Create and activate a virtual environment

4 On Windows:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate

5 On macOS/Linux:
    ```bash
    python3 -m venv venv
    source venv/bin/activate

6 Install dependencies
    ```bash
    pip install -r requirements.txt

7 Run the Python script stock_prediction_lstm.py
    ```bash
    python stock_prediction_lstm.py



author suvigya
