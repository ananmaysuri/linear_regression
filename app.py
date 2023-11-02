import json
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

def model(file):
    df = pd.read_csv(file)

    # Data Preprocessing
    start_date = datetime(2021, 1, 1)
    df['# Date'] = pd.to_datetime(df['# Date'])
    df['Day_Number'] = (df['# Date'] - start_date).dt.days
    df['Month'] = df['# Date'].dt.month

    # Convert to PyTorch tensors
    X = torch.tensor(df['Day_Number'].values, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(df['Receipt_Count'].values, dtype=torch.float32).view(-1, 1)

    # Normalize both X and y
    X_min, X_max = X.min(), X.max()
    y_min, y_max = y.min(), y.max()

    X_normalized = (X - X_min) / (X_max - X_min)
    y_normalized = (y - y_min) / (y_max - y_min)

    # Linear Regression Model
    class LinearRegressionModel(torch.nn.Module):
        def __init__(self):
            super(LinearRegressionModel, self).__init__()
            self.weights = torch.nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))
            self.bias = torch.nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))

        def forward(self, x):
            return x * self.weights + self.bias

    # Model, Loss Function, Optimizer
    model = LinearRegressionModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training Loop
    epochs = 5000
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_normalized)
        loss = criterion(outputs, y_normalized)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 500 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Aggregation of 2021 Data
    monthly_totals_2021 = df.groupby('Month')['Receipt_Count'].sum()

    # Prediction and Aggregation for 2022
    # 15th of each month would be having the approximate average receipt count of that month
    months_2022 = [datetime(2022, month_number, 15) for month_number in range(1, 13)]
    day_numbers_2022 = [(date - start_date).days for date in months_2022]
    X_2022 = torch.tensor(day_numbers_2022, dtype=torch.float32).view(-1, 1)
    X_2022_normalized = (X_2022 - X_min) / (X_max - X_min)

    predictions_2022_normalized = model(X_2022_normalized).detach().numpy().flatten()

    # Convert y_min and y_max to NumPy arrays for de-normalization
    y_min_np = y_min.numpy()
    y_max_np = y_max.numpy()

    predictions_2022 = (predictions_2022_normalized * (y_max_np - y_min_np) + y_min_np).astype(int)

    days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    monthly_totals_2022 = predictions_2022 * days_in_month
    series_monthly_totals_2022 = pd.Series(monthly_totals_2022, index=range(1, 13))

    # Print Monthly Totals
    print(f"2021 Monthly Totals: {monthly_totals_2021.values}")
    print(f"2022 Monthly Totals: {monthly_totals_2022}")

    plt.figure(figsize=(12, 6))
    plt.scatter(df['Day_Number'], df['Receipt_Count'], label='2021 Data')
    plt.scatter(day_numbers_2022, predictions_2022, color='red', label='2022 Predictions')
    plt.xlabel('Day Number')
    plt.ylabel('Receipt Count')
    plt.title('Receipt Count Prediction')
    plt.legend()
    plt.savefig('static/predictions.png')

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_totals_2021.index, monthly_totals_2021.values, label='2021 Total per Month', marker='o')
    plt.plot(series_monthly_totals_2022.index, series_monthly_totals_2022.values, color = 'red', label='2022 Predicted Total per Month', marker='o')
    plt.xlabel('Month')
    plt.ylabel('Total Receipt Count')
    plt.title('Monthly Receipt Count Totals')
    plt.xticks(range(1, 13))
    plt.legend()
    plt.savefig('static/monthly_totals.png')

    months = ["January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"]

    # Zip month names with values and convert to dictionary
    month_values = dict(zip(months, monthly_totals_2022.tolist()))

    return month_values

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No file selected for uploading'}), 400

    if file and file.filename.endswith('.csv'):
        try:
            output = model(file)
            # Convert the dictionary to JSON
            json_data = json.dumps(output)
            return jsonify({'message': 'File successfully uploaded, read and inferenced', 'data': json_data}), 200
        except Exception as e:
            return jsonify({'message': 'Error processing file', 'error': str(e)}), 500

    else:
        return jsonify({'message': 'Allowed file type is csv'}), 400

@app.route('/predict',methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No file selected for uploading'}), 400

    if file and file.filename.endswith('.csv'):
        try:
            output = model(file)
            return render_template("home.html", output_dict=output)
        except Exception as e:
            return jsonify({'message': 'Error processing file', 'error': str(e)}), 500

    else:
        return jsonify({'message': 'Allowed file type is csv'}), 400

if __name__=="__main__":
    app.run(debug=True)