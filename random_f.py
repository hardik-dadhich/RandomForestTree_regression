# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:41:05 2018

@author: DeLL
"""
# importing necessary library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

#fitting Random forest Regrasssion to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 400, random_state= 0)
regressor.fit(X, y)

#predicting the new result
y_pred = regressor.predict(6.5)

#visulising the regression result
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color= 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random forest regression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()