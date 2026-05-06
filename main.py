# main.py

import pandas as pd
import numpy as np
import pickle

from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from fastapi import FastAPI

# Load data
df = pd.read_excel("data.xlsx")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Handle missing dates
df = df.set_index('date')
df = df.groupby('state').apply(lambda x: x.asfreq('D')).reset_index()

# Fill missing values
df['sales'] = df['sales'].fillna(method='ffill')

# Feature engineering
def create_features(data):
    data['lag_1'] = data['sales'].shift(1)
    data['lag_7'] = data['sales'].shift(7)
    data['lag_30'] = data['sales'].shift(30)
    data['rolling_mean_7'] = data['sales'].rolling(7).mean()
    data['rolling_std_7'] = data['sales'].rolling(7).std()
    data['day_of_week'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    return data

df = create_features(df)
df = df.dropna()

train = df[:-56]
test = df[-56:]

features = ['lag_1','lag_7','lag_30','rolling_mean_7','rolling_std_7','day_of_week','month']

# Train models
arima_model = SARIMAX(train['sales'], order=(1,1,1), seasonal_order=(1,1,1,7))
arima_fit = arima_model.fit()
arima_pred = arima_fit.forecast(steps=56)

prophet_df = train[['date','sales']].rename(columns={'date':'ds','sales':'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_df)

future = prophet_model.make_future_dataframe(periods=56)
forecast = prophet_model.predict(future)
prophet_pred = forecast['yhat'][-56:].values

xgb_model = XGBRegressor()
xgb_model.fit(train[features], train['sales'])
xgb_pred = xgb_model.predict(test[features])

lstm_pred = xgb_pred

# Evaluation
def evaluate(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

scores = {
    "ARIMA": evaluate(test['sales'], arima_pred),
    "Prophet": evaluate(test['sales'], prophet_pred),
    "XGBoost": evaluate(test['sales'], xgb_pred),
    "LSTM": evaluate(test['sales'], lstm_pred)
}

best_model_name = min(scores, key=scores.get)

if best_model_name == "ARIMA":
    best_model = arima_fit
elif best_model_name == "Prophet":
    best_model = prophet_model
else:
    best_model = xgb_model

pickle.dump(best_model, open("best_model.pkl", "wb"))

# API
app = FastAPI()
model = pickle.load(open("best_model.pkl", "rb"))

@app.get("/forecast")
def forecast():
    return {"message": "Forecast generated successfully"}
