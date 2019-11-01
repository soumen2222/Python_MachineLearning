import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score


HouseData = pd.read_csv("kc_house_data.csv")
trainData = pd.read_csv("kc_house_train_data.csv")
testData = pd.read_csv("kc_house_test_data.csv")

regr = linear_model.LinearRegression()
x = np.asanyarray(trainData[['sqft_living']])
y = np.asanyarray(trainData[['price']])
regr.fit (x, y)
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


test_x = np.asanyarray(testData[['sqft_living']])
test_y = np.asanyarray(testData[['price']])

print(regr.predict(2650))
test_y_hat = regr.predict(test_x)

print("Residual sum of squares (MSE): ", np.sum((test_y_hat - test_y) ** 2))


regr = linear_model.LinearRegression()
x = np.asanyarray(trainData[['bedrooms']])
y = np.asanyarray(trainData[['price']])
regr.fit (x, y)
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


test_x = np.asanyarray(testData[['bedrooms']])
test_y = np.asanyarray(testData[['price']])
test_y_hat = regr.predict(test_x)

print("Residual sum of squares (MSE): ", np.sum((test_y_hat - test_y) ** 2))