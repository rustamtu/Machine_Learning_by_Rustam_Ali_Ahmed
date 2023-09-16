# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 10:33:33 2023

@author: Rustam Ali Ahmed
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 10:12:12 2023

@author: Rustam Ali Ahmed
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

print('hello')

dataset=pd.read_csv('dataset\DiabetesHealthIndicatorsDataset\data.csv')

# x is 2nd to last col
X = dataset.iloc[:, 1:].values 
# y is first col
y = dataset.iloc[:, 0:1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(X_train)
x_test=sc_x.fit_transform(X_test)

print('After fitting')
print('x_train')
print(x_train)
print('y_train')
print(y_train)

len(x_train)
len(x_test)