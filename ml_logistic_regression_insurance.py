# -*- coding: utf-8 -*-
"""
ml_logistic_regression
Created on Sat Sep 16 10:33:33 2023

@author: Rustam Ali Ahmed
https://www.youtube.com/watch?v=O0Ka_nBRtN0
04:14:10 - Logistic Regression (Binary Classification) | Machine Learning Tutorial
https://www.youtube.com/watch?v=O0Ka_nBRtN0&t=15250s

to predict regions
dataset['region'].replace({'southwest':'0', 'southeast':'1', 'northwest':'2', 'northeast':'3'}, inplace=True)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset=pd.read_csv('dataset\insurance.csv')
# age,sex,bmi,children,smoker,region,charges
print(dataset)
dataset['sex'].unique()
dataset['smoker'].unique()
dataset['region'].unique()

dataset['sex'].replace({'female':'0', 'male':'1'}, inplace=True)
dataset['region'].replace({'southwest':'0', 'southeast':'1', 'northwest':'2', 'northeast':'3'}, inplace=True)
dataset['smoker'].replace({'yes':'0', 'no':'1'}, inplace=True)
print(dataset)

X=dataset[['age','sex','bmi','children','smoker','charges']]
y=dataset['region']
# plt.scatter(x='Inputs', y='charges', data=[X, y])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(len(X_train))
print(len(X_test))

print(len(y_train))
print(len(y_test))

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train, y_train.values.ravel())
print(X_test[0:10])
print(lr.predict(X_test[0:10]))
print(y_test[0:10])

import seaborn as sns
sns.pairplot(dataset[['age','sex','bmi','children','smoker','charges', 'region']], hue='region')

