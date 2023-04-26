#libraries importing
import numpy as np
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import pickle

from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries, window=12, cutoff=0.01):
    # Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()
    dftest = adfuller(timeseries, autolag='AIC', maxlag=2)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    pvalue = dftest[1]
    if pvalue < cutoff:
        return 1
    else:
        return 0

def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)

def FeatureEngineering(df,x_name,y_name,typeData = 3):
  '''
  typeData: 0-> Day 1->Month 2->Weakly 3-> Hourly
  '''
  '''df = df[[x_name,y_name]]
  df = df.set_index(x_name)
  df.index = pd.to_datetime(df.index)
  df = df.dropna()'''

  res = {"Stationarity":test_stationarity(df),"ACF":np.mean(acf(df,nlags=2)),"PACF":np.mean(pacf(df,nlags=2)),"YMean":df[y_name].mean(),"TypeData":typeData,"YStd":df[y_name].std(),"CosineYMean": np.cos(df[y_name]).mean(),"SineYMean":np.sin(df[y_name]).mean(),"MaxY":max(df[y_name]),"MinY":min(df[y_name]),"MedianY":df[y_name].median(),"25PercY":df[y_name].quantile(q=0.25),"75PercY":df[y_name].quantile(q=0.75),"RBFY":rbf(2*df[y_name],df[y_name],1).mean(), }
  cal = calendar()
  holidays = cal.holidays(start=df.index.min(), end=df.index.max())
  res["NoOfHolidays"] = df.index.isin(holidays).sum()

  return res


def FE(typeData):
    df = pd.read_csv("result.csv")
    df.set_index("point_timestamp",inplace=True)
    df.index = pd.to_datetime(df.index)
    df.dropna(inplace=True)

    x_name, y_name = "point_timestamp", "point_value"
    res = None

    if typeData.lower() == 'daily':
        res = FeatureEngineering(df, x_name, y_name, typeData=0)

    elif typeData.lower() == 'monthly':
        res = FeatureEngineering(df, x_name, y_name, typeData=1)

    elif typeData.lower() == 'weekly':
        res = FeatureEngineering(df, x_name, y_name, typeData=2)

    elif typeData.lower() == 'hourly':
        res = FeatureEngineering(df, x_name, y_name, typeData=3)
    #print(res)
    res = [res]
    res = pd.DataFrame.from_dict(res)
    #print(res)
    loaded_model = pickle.load(open("classifier.pkl", 'rb'))
    output = loaded_model.predict(res)
    print(output)

    model = {0:"ARIMA",1:"SARIMA",2:"XGBOOST"}

    return model[output[0]]
