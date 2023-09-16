# -*- coding: utf-8 -*-
"""
ml_linear_regression_multi_var_BostonHousing
Created on Sat Sep 16 10:33:33 2023

@author: Rustam Ali Ahmed
https://www.youtube.com/watch?v=O0Ka_nBRtN0
03:37:13 - Linear Regression Multiple Variable | Machine Learning Tutorial
https://www.youtube.com/watch?v=O0Ka_nBRtN0&t=13033s
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

dataset=pd.read_csv('dataset\BostonHousing.csv')
# "crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat","medv"
print(dataset)
X=dataset[["crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat"]]
y=dataset['medv']
reg=linear_model.LinearRegression()
# reg.fit(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
reg.fit(X_train, y_train)
sns.lmplot(x='age', y='medv', data=dataset)

# y=ax+by+cz+.........+d
# coef_=[a,b,c,.....]
# intercept_ = d
print('Coeficiant: ', reg.coef_)
print('intercept_: ', reg.intercept_)

print(X_test[0:1])
print(reg.predict(X_test[0:1]))
# reg.predict(X_test.values.ravel())
print(y_test[0:1])
print('Accuracy: ', reg.score(X_test, y_test)*100, '%')



# dataset=dataset_original #dataset_original[["Age","Diabetes_binary"]]

# 100 rows
# dataset=dataset[1:100]



# X=dataset[['age']]
# y=dataset[['medv']]
# # # plt.scatter(x='Inputs', y='charges', data=[X, y])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(len(X_train))
# print(len(X_test))

# print(len(y_train))
# print(len(y_test))

# from sklearn.linear_model import LogisticRegression
# lr=LogisticRegression()
# lr.fit(X_train, y_train.values.ravel())
# # print(X_test)
# # print(lr.predict(X_test))




