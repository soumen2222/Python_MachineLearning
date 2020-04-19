import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score


sales = pd.read_csv("Philadelphia_Crime_Rate_noNA.csv")
print(sales.head(5))

cdf = sales[['HousePrice','CrimeRate','MilesPhila']]

plt.scatter(cdf.CrimeRate, cdf.HousePrice,  color='blue')
plt.xlabel("CrimeRate")
plt.ylabel("HousePrice")
plt.show()

msk = np.random.rand(len(sales)) < 0.8
train = sales[msk]
test = sales[~msk]

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['CrimeRate']])
y = np.asanyarray(train[['HousePrice']])
regr.fit (x, y)
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

plt.scatter(train.CrimeRate, train.HousePrice,  color='blue')
plt.plot(x, regr.coef_[0][0]*x + regr.intercept_[0], '-r')
plt.xlabel("CrimeRate")
plt.ylabel("HousePrice")
plt.show()



sales_noCC = sales[sales['MilesPhila'] != 0.0]
plt.scatter(sales_noCC.CrimeRate, sales_noCC.HousePrice,  color='blue')
plt.xlabel("CrimeRate")
plt.ylabel("HousePrice")
plt.show()

msk = np.random.rand(len(sales_noCC)) < 0.8
train = sales_noCC[msk]
test = sales_noCC[~msk]

regr_NOCC = linear_model.LinearRegression()
x = np.asanyarray(train[['CrimeRate']])
y = np.asanyarray(train[['HousePrice']])
regr.fit (x, y)
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

plt.scatter(train.CrimeRate, train.HousePrice,  color='blue')
plt.plot(x, regr.coef_[0][0]*x + regr.intercept_[0], '-r')
plt.xlabel("CrimeRate")
plt.ylabel("HousePrice")
plt.show()