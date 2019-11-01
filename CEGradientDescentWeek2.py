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

def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1 # add a constant column to an SFrame
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    # select the columns of data_SFrame given by the ‘features’ list into the SFrame ‘features_sframe’

    for i in features:
        features_sframe[i] = data_sframe[i]
    # this will convert the features_sframe into a numpy matrix with GraphLab Create >= 1.7!!
    features_matrix = features_sframe.to_numpy()
    # assign the column of data_sframe associated with the target to the variable ‘output_sarray’

    # this will convert the SArray into a numpy array:
    output_array = output_sarray.to_numpy() # GraphLab Create>= 1.7!!
    return(features_matrix, output_array)

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

