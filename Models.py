import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def arima_(df,date):
  from statsmodels.tsa.arima.model import ARIMA
  from sklearn.metrics import mean_absolute_percentage_error as mape
  from itertools import product
  # Define the range of values for p, d, and q
  p = range(0, 5)
  d = range(0, 2)
  q = range(0, 3)

  # Generate all possible combinations of p, d, and q values
  pdq = list(product(p, d, q))

  # Define an empty list to store the AIC values for each combination of hyperparameters
  aic_values = []

  # Iterate over all combinations of hyperparameters and fit an ARIMA model for each combination
  for params in pdq:
      try:
          model = ARIMA(df, order=params)
          results = model.fit()
          aic_values.append((params, results.aic))
      except:
          continue

  # Print the combination of hyperparameters that gives the lowest AIC value
  best_params, best_aic = min(aic_values, key=lambda x: x[1])

  train, test = df[:int(len(df)*0.9)], df[int(len(df)*0.9):]
  model = ARIMA(train, order=best_params).fit()
  arima_prediction = model.forecast(steps=len(test))
  date = pd.to_datetime(date)
  pred = model.predict(start=date)
  return mape(test, arima_prediction.values)

def sarimax_(df,date):
  from statsmodels.tsa.statespace.sarimax import SARIMAX
  from sklearn.metrics import mean_absolute_percentage_error as mape
  from itertools import product
  # Define the range of values for p, d, and q

  train, test = data[:int(len(df)*0.9)], data[int(len(df)*0.9):]
  model1 = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
  #dates= pd.date_range(start=train.index[len(train)-1], periods=len(test))
  sarima_prediction = model1.forecast(steps=len(test))
  date = pd.to_datetime(date)
  pred = model1.predict(start=date)
  return mape(test, sarima_prediction.values),pred


def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df = pd.DataFrame(df)
    df['date'] = df.index
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear

    X = df[['dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear']]
    if label:
        y = df[label]

        return X, y
    return X
def feature_selection_pred(date):
    temp = date.split("-")
    from datetime import date
    d = date(int(temp[0]), int(temp[1]), int(temp[2]))

    dayofweek= d.weekday()
    quarter = int(temp[1])%4
    month = int(temp[1])
    year = int(temp[0])
    dayofyear = int((d - date(d.year, 1, 1)).days + 1)
    dayofmonth = int(d.day)
    week_of_year_tuple = d.isocalendar()
    weekofyear = int(week_of_year_tuple[1])

    X = [[dayofweek, quarter, month, year,
            dayofyear, dayofmonth, weekofyear]]

    return X
def xgboost_(df,date):
  import xgboost as xgb
  from xgboost import plot_importance, plot_tree
  from sklearn.metrics import mean_absolute_percentage_error as mape
  data = df.copy()
  xgdata = create_features(data,'point_value')
  Xtrain,Ytrain, Xtest,Ytest = xgdata[0][:int(len(xgdata[0])*0.9)],xgdata[1][:int(len(xgdata[1])*0.9)], xgdata[0][int(len(xgdata[0])*0.9):],xgdata[1][int(len(xgdata[1])*0.9):]
  reg = xgb.XGBRegressor(n_estimators=10)
  reg.fit(Xtrain, Ytrain,
          verbose=False)
  xgforecast = reg.predict(Xtest)

  pred_features = feature_selection_pred(date)
  print(pred_features,len(pred_features))
  pred = reg.predict(np.array(pred_features[0]).reshape(1,-1))
  return mape(xgforecast, Ytest),pred

def Model(type,df,date):
    res = None
    if type == "ARIMA":
        res = arima_(df,date)
    elif type == "SARIMA":
        res = sarimax_(df,date)
    else:
        res = xgboost_(df,date)

    return res
