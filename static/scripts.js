document.getElementById('stock-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const stockSymbol = document.getElementById('stock_symbol').value;

    fetch(`/predict?stock_symbol=${stockSymbol}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('error-message').innerText = 'Error: ' + data.error;
                document.getElementById('error-message').classList.remove('hidden');
                document.getElementById('output').classList.add('hidden');
            } else {
                document.getElementById('error-message').classList.add('hidden');
                document.getElementById('stock-symbol-display').innerText = stockSymbol;
                document.getElementById('predicted-price').innerText = data.predicted_price;

                const stockPriceData = data.stockPriceData;
                const historicalData = data.historicalData;

                new Chart(document.getElementById('stock-price-chart'), {
                    type: 'line',
                    data: {
                        labels: stockPriceData.labels,
                        datasets: [{
                            label: 'Predicted Stock Price',
                            data: stockPriceData.values,
                            borderColor: '#4CAF50',
                            fill: false
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Time'
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Price'
                                }
                            }
                        }
                    }
                });

                new Chart(document.getElementById('historical-chart'), {
                    type: 'line',
                    data: {
                        labels: historicalData.labels,
                        datasets: [{
                            label: 'Historical Stock Price',
                            data: historicalData.values,
                            borderColor: '#2196F3',
                            fill: false
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Time'
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Price'
                                }
                            }
                        }
                    }
                });

                document.getElementById('output').classList.remove('hidden');
            }
        })
        .catch(error => {
            document.getElementById('error-message').innerText = 'Error: ' + error;
            document.getElementById('error-message').classList.remove('hidden');
            document.getElementById('output').classList.add('hidden');
        });
});
