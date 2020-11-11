# coding=<utf-8>
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# -*- coding: utf-8 -*-
hw = pd.read_csv('C:\datamining\project\전년도_65세이상인구비율2.csv', encoding='CP949')
hw = hw.fillna(0)

X = hw.iloc[:, 2:-1]
Y = hw.iloc[:, -1:]

year=np.array(range(2011,2021))

exam = hw.iloc[1:2, 2:]
exam=np.array(exam)

exam=exam.reshape(-1,1)
year=year.reshape(-1,1)

print(exam.shape)
print(year.shape)

X = np.array(X)
Y = np.array(Y)
print(X.shape)
Y = Y.reshape(-1,1)
print(Y.shape)

line_fit = LinearRegression()
line_fit.fit(year,exam)
y_predicted =line_fit.predict(year)

print('기울기 :',line_fit.coef_) #기울기
print('y 절편 :',line_fit.intercept_) # 절편

plt.plot(year,exam,'o')
plt.plot(year,y_predicted)
plt.show()