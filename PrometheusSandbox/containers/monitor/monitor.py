import json
import pandas as pd
import requests
import time
from prophet import Prophet
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from prometheus_client import start_http_server, Gauge, Counter, Summary

# Prometheus metrics
start_http_server(8099)  # Start Prometheus metrics server
anomaly_count = Gauge('anomaly_count', 'Number of anomalies detected')
mae_metric = Gauge('mae', 'Mean Absolute Error')
mape_metric = Gauge('mape', 'Mean Absolute Percentage Error')

# Define parameters for the number of minutes in train and test data URLs
train_minutes = 1  # Number of minutes for training data
test_minutes = 5   # Number of minutes for test data

# Prometheus query URLs for train and test data
train_query_url = f'http://prometheus:9090/api/v1/query?query=request_time_train[{train_minutes}m]'
test_query_url = f'http://prometheus:9090/api/v1/query?query=request_time_test[{test_minutes}m]'

# Define the forecast count (number of times to predict before getting new train data)
forecast_count = 2
forecast_iteration = 0  # Initialize forecast iteration counter

# Create an empty DataFrame to store forecast results
forecast_results = pd.DataFrame(columns=['Timestamp', 'Anomalies', 'MAE', 'MAPE'])

print(f'Forecast_count : {forecast_count}' )
print(f'Train minutes : {train_minutes}' )
print(f'Test minutes : {test_minutes}' )


while True:
    try:
        if forecast_iteration % forecast_count == 0:
            # Fetch new training data only when forecast_iteration is a multiple of forecast_count
            print()
            print('Fetching new training data:')
            print(train_query_url)

            # Fetch training data from Prometheus for the specified number of minutes
            train_response = requests.get(train_query_url)
            train_data = train_response.json()

            # Preprocess training data
            df_train = pd.DataFrame(train_data['data']['result'][0]['values'])
            df_train.columns = ['ds', 'y']
            df_train['ds'] = df_train['ds'].apply(lambda sec: datetime.fromtimestamp(sec))
            df_train['y'] = df_train['y'].astype(float)

            # Initialize Prophet model
            model = Prophet(interval_width=0.99, yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)

            # Fit the model on the training dataset
            model.fit(df_train)

        # Print fetching of test data
        print(f'Sleeping for 1 minute before fetching test data (Iteration: {forecast_iteration})')
        time.sleep(60)
        print('Fetching test data:')
        print(test_query_url)

        # Fetch test data from Prometheus for the specified number of minutes
        test_response = requests.get(test_query_url)
        test_data = test_response.json()

        # Preprocess test data
        df_test = pd.DataFrame(test_data['data']['result'][0]['values'])
        df_test.columns = ['ds', 'y']
        df_test['ds'] = df_test['ds'].apply(lambda sec: datetime.fromtimestamp(sec))
        df_test['y'] = df_test['y'].astype(float)

        # Make predictions on the test dataset
        forecast = model.predict(df_test)

        # Merge actual and predicted values
        performance = pd.merge(df_test, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

        # Calculate MAE and MAPE
        performance_MAE = mean_absolute_error(performance['y'], performance['yhat'])
        performance_MAPE = mean_absolute_percentage_error(performance['y'], performance['yhat'])

        # Print results
        print('Results:')
        print(f'The MAE for the model is {performance_MAE}')
        print(f'The MAPE for the model is {performance_MAPE}')

        # Set the Prometheus Gauge metrics for MAE and MAPE
        mae_metric.set(performance_MAE)
        mape_metric.set(performance_MAPE)

        # Create an anomaly indicator
        performance['anomaly'] = performance.apply(lambda row: 1 if (row['y'] < row['yhat_lower'] or row['y'] > row['yhat_upper']) else 0, axis=1)

        # Check the number of anomalies
        anomaly_count.set(performance['anomaly'].sum())  # Set the Prometheus gauge metric

        # Append forecast results to the DataFrame
        forecast_results = forecast_results.append({'Timestamp': datetime.now(), 'Anomalies': performance['anomaly'].sum(), 'MAE': performance_MAE, 'MAPE': performance_MAPE}, ignore_index=True)

        forecast_iteration += 1  # Increment forecast iteration counter

    except Exception as e:
        print(f'Error: {str(e)}')

    # Sleep for 60 seconds before the next iteration
    time.sleep(60)

# Print the final forecast results DataFrame
print()
print('\nFinal Forecast Results:')
print(forecast_results)
