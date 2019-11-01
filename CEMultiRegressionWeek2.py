import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
HouseData = pd.read_csv("kc_house_data.csv", dtype=dtype_dict)
trainData = pd.read_csv("kc_house_train_data.csv", dtype=dtype_dict)
testData = pd.read_csv("kc_house_test_data.csv", dtype=dtype_dict)

trainData['bedrooms_squared'] = trainData['bedrooms']*trainData['bedrooms']
testData['bedrooms_squared'] = testData['bedrooms']*testData['bedrooms']

trainData['bed_bath_rooms'] = trainData['bedrooms']*trainData['bathrooms']
testData['bed_bath_rooms'] = testData['bedrooms']*testData['bathrooms']

trainData['log_sqft_living'] = np.log(trainData['sqft_living'])
testData['log_sqft_living'] = np.log(testData['sqft_living'])

trainData['lat_plus_long'] = trainData['lat'] + trainData['long']
testData['lat_plus_long'] = testData['lat'] + testData['long']

"""
Squaring bedrooms will increase the separation between not many bedrooms (e.g. 1) and lots of bedrooms (e.g. 4) since 1^2 = 1 but 4^2 = 16. Consequently this variable will mostly affect houses with many bedrooms.
Bedrooms times bathrooms is what's called an "interaction" variable. It is large when both of them are large.
Taking the log of square feet has the effect of bringing large values closer together and spreading out small values.
Adding latitude to longitude is non-sensical but we will do it anyway

"""

# mean of the specific column
print("bedrooms_squared_mean:",testData.loc[:,"bedrooms_squared"].mean())
print("bed_bath_rooms_mean:",testData.loc[:,"bed_bath_rooms"].mean())
print("log_sqft_living_mean:",testData.loc[:,"log_sqft_living"].mean())
print("lat_plus_long_mean:",testData.loc[:,"lat_plus_long"].mean())

# Create Multiple Regression Model

regr1 = linear_model.LinearRegression()
x = np.asanyarray(trainData[['sqft_living','bedrooms','bathrooms','lat','long']])
y = np.asanyarray(trainData[['price']])
regr1.fit (x, y)
print ('Coefficients_Reg1: ', regr1.coef_)
print ('Intercept_Reg1: ',regr1.intercept_)


test_x = np.asanyarray(trainData[['sqft_living','bedrooms','bathrooms','lat','long']])
test_y = np.asanyarray(trainData[['price']])
test_y_hat = regr1.predict(test_x)

print("Residual sum of squares (MSE)_Reg1: ", np.sum((test_y_hat - test_y) ** 2))


regr2 = linear_model.LinearRegression()
x = np.asanyarray(trainData[['sqft_living','bedrooms','bathrooms','lat','long','bed_bath_rooms']])
y = np.asanyarray(trainData[['price']])
regr2.fit (x, y)
print ('Coefficients_Reg2: ', regr2.coef_)
print ('Intercept_Reg2: ',regr2.intercept_)


test_x = np.asanyarray(trainData[['sqft_living','bedrooms','bathrooms','lat','long','bed_bath_rooms']])
test_y = np.asanyarray(trainData[['price']])
test_y_hat = regr2.predict(test_x)

print("Residual sum of squares (MSE)_Reg2: ", np.sum((test_y_hat - test_y) ** 2))


regr3 = linear_model.LinearRegression()
x = np.asanyarray(trainData[['sqft_living','bedrooms','bathrooms','lat','long','bed_bath_rooms','bedrooms_squared','log_sqft_living','lat_plus_long']])
y = np.asanyarray(trainData[['price']])
regr3.fit (x, y)
print ('Coefficients_Reg3: ', regr3.coef_)
print ('Intercept_Reg3: ',regr3.intercept_)

test_x = np.asanyarray(trainData[['sqft_living','bedrooms','bathrooms','lat','long','bed_bath_rooms','bedrooms_squared','log_sqft_living','lat_plus_long']])
test_y = np.asanyarray(trainData[['price']])
test_y_hat = regr3.predict(test_x)

print("Residual sum of squares (MSE)_Reg3: ", np.sum((test_y_hat - test_y) ** 2))

