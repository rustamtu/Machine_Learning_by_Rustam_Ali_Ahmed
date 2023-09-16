# -*- coding: utf-8 -*-
"""
ml_multiclass_classification_BostonHousing
Created on Sat Sep 16 10:33:33 2023

@author: Rustam Ali Ahmed
https://www.youtube.com/watch?v=O0Ka_nBRtN0
04:14:10 - Logistic Regression (Binary Classification) | Machine Learning Tutorial
https://www.youtube.com/watch?v=O0Ka_nBRtN0&t=15250s
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset=pd.read_csv('dataset\BostonHousing.csv')
# "crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat","medv"

# dataset=dataset_original #dataset_original[["Age","Diabetes_binary"]]

# 100 rows
# dataset=dataset[1:100]

print(dataset)

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




