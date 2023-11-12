# Timely

## Problem Statement:
Build a classification model from the sample data provided to predict which
time series model can be a best fit to the new csv data that is given as input and predict
the output for a particular date given as input from the time series model chosen by our
classifier model ( Classifier Model should be build based on MAPE value of the time
series models).
## Data:
The sample dataset given has 4 categories Daily , Hourly , Weekly , Monthly .
There are totally 36 datasets available in sample data . Each contains Two columns
‘point_timestamp’ , ‘point_value’ . point_timestamp is the timestamp of the data and the
point_value is an integer . The data is preprocessed by removing any missing values
and scaling it to have a zero mean and unit variance.
## Models:
The time series models which were used here are ARIMA , SARIMA , XGboost.
ARIMA:The order of the ARIMA model is denoted as (p, d, q), where p is the order of
the autoregressive component, d is the degree of differencing, and q is the order of the
moving average component. The ARIMA model can be used to forecast future values of
a time series, as well as to analyze the patterns and trends in the data.
SARIMA:
SARIMA, which stands for Seasonal Autoregressive Integrated Moving Average, is
a time series forecasting model that extends the ARIMA model by incorporating
seasonal components.
The SARIMA model can be written as ARIMA(p, d, q)(P, D, Q)s, where p, d, and
q are the non-seasonal components of the ARIMA model, and (P, D, Q)s are the
seasonal components.
XGboost:
XGBoost (Extreme Gradient Boosting) is a popular machine learning algorithm
that can be used for time series forecasting. XGBoost is a tree-based ensemble method that builds a series of decision trees and combines their predictions to generate a final
forecast.
For XGboost model various features are extracted from the given data such as
● Dayofweek
● Month
● Year
● Dayofyear
● Dayofmonth
● Weekofyear
These are the features that are used for training an xgboost model and also for the the
training of the sample data lags are also used as feature for the building of xgboost
model.

## Performance measure:
MAPE:MAPE stands for Mean Absolute Percentage Error. It is a commonly
used metric for measuring the accuracy of predictions made by forecasting models.
MAPE calculates the average absolute percentage difference between the predicted
values and the actual values

## Classifier Model Building:
We need to build a classifier model to choose the best model for the
new input data given . For building the classifier model we need to extract various
features from the sample data given . some of the features are
● Stationarity
● ACF
● PACF
● Ymean
● YSTD
● MAXy
● Miny
● MedianY
● Percentile
With these features as X and label should be done by taking the minimum of the MAPE
value from all the 3 models and then the label will be assigned . With this we will get a
dataframe of multiway classification
For the train data the accuracy was 100% and for the test data the accuracy is 75%
Now with the data available an classification model XGboost is done and save the
model as a pickle file

## Finding the Best Model:
After the building of a classification model ,with the help of that model our
best model for forecasting is predicted by extracting the features from the input data and
predict the best model using the pickle file which was already done
## Forecasting:
After finding the best model , the input data is trained in the model that was
chosen as the best model and the MAPE value is calculated for that data and also the
point_value is forecasted for the given particular timestamp
## Fast API:
FastAPI is a modern, fast (hence the name) web framework for building APIs
with Python. It is built on top of the Starlette framework, which is a lightweight ASGI
(Asynchronous Server Gateway Interface) framework, and uses Python 3.6+ type hints
to provide fast, efficient, and easy-to-use routing and data validation.
User interface:HTML (Hypertext Markup Language) is used to create the structure
and content of web pages. HTML provides a set of tags that are used to define the
various elements of a web page, including text, images, links, forms, and other
interactive elements. Here is a brief overview of how to create a simple user interface
using HTML

## Project Workflow:
![image](https://user-images.githubusercontent.com/69851775/234507163-882ca4ae-24dd-46d0-a78f-b5e48a370426.png)

## Demo
![image](https://user-images.githubusercontent.com/69851775/234507306-14be61f1-174c-41d9-8853-7da9b3de7e62.png)
![image](https://user-images.githubusercontent.com/69851775/234507349-757da2a2-bb7a-444a-95bf-bcc715d59d57.png)
![image](https://user-images.githubusercontent.com/69851775/234507442-af22a7d5-fcaa-49a2-a374-b7961f84cbfa.png)




 
