import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# Load dataset from CSV file
file_path = 'ett/ETTh2.csv'  # Update with your CSV file path
df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')

# Extracting date and OT columns
df_ot = df[['OT']]

# Split data into training and testing (80/20 split)
train_size = int(len(df_ot) * 0.8)
train, test = df_ot[:train_size], df_ot[train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(5, 1, 0))  # order=(p, d, q)
model_fit = model.fit()

# Make forecast
forecast = model_fit.forecast(steps=len(test))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(train, label='Training Data')
plt.plot(test, label='Test Data')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.xlabel('Date')
plt.ylabel('OT')
plt.title('ARIMA Model - Forecasting OT')
plt.legend()
plt.show()