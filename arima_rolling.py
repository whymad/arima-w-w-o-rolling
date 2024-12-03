import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import itertools
import warnings
from sklearn.metrics import mean_squared_error
from scipy.stats import boxcox
import numpy as np

# Load dataset from CSV file
file_path = 'ett/ETTh2.csv'  # Update with your CSV file path
df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')

# Extracting date and OT columns
df_ot = df[['OT']]

# Ensure data is positive by shifting if necessary
if (df_ot['OT'] <= 0).any():
    min_value = abs(df_ot['OT'].min()) + 1
    df_ot.loc[:, 'OT'] = df_ot['OT'] + min_value

# Apply Box-Cox Transformation to stabilize variance
df_ot.loc[:, 'OT'], lam = boxcox(df_ot['OT'])

# Split data into training and testing (80/20 split)
train_size = int(len(df_ot) * 0.8)
train, test = df_ot[:train_size], df_ot[train_size:]

# Rolling Forecasts using ARIMA with optimized parameters
history = [x for x in train['OT']]
y = test['OT']

# Make first prediction
predictions = list()
model = ARIMA(history, order=(1, 1, 1))
model_fit = model.fit()
yhat = model_fit.forecast()[0]
predictions.append(yhat)
history.append(y.iloc[0])

# Rolling forecasts
for i in range(1, len(y)):
    # Predict
    model = ARIMA(history, order=(1, 1, 1))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    # Append prediction
    predictions.append(yhat)
    # Append observation to history
    obs = y.iloc[i]
    history.append(obs)

# Calculate and print RMSE
rmse = mean_squared_error(test, predictions, squared=False)
print(f'RMSE: {rmse}')

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Test Data')
plt.plot(test.index, predictions, label='Rolling Forecast', color='red')
plt.xlabel('Date')
plt.ylabel('OT')
plt.title('ARIMA Model - Rolling Forecasting OT')
plt.legend()
plt.show()

