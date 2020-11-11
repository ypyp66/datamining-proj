import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

hw = pd.read_csv('C:\datamining\project\전년도_65세이상인구비율2.csv', encoding='CP949')
hw = hw.fillna(0)

X = hw.iloc[:, 2:-1] #2011~2019년 데이터
Y = hw.iloc[:, -1:] #2020년 데이터

line_fitter = LinearRegression()
line_fitter.fit(X, Y)
y_predicted = line_fitter.predict(X)
print(y_predicted.shape) #263,1

year = ['2011년 65세이상 인구 비율', '2012년 65세이상 인구 비율',
        '2013년 65세이상 인구 비율', '2014년 65세이상 인구 비율',
        '2015년 65세이상 인구 비율', '2016년 65세이상 인구 비율',
        '2017년 65세이상 인구 비율', '2018년 65세이상 인구 비율',
        '2019년 65세이상 인구 비율']
#year = ['2011','2012','2013','2014','2015','2016','2017','2018','2019']
pred = []

for i in year:
    print(hw.filter(year).shape)

location = input('지역 입력:')
print(type(location))
df = hw[hw['행정구역'].str.contains(location)]

X_ = df.iloc[:,2:] #2011~2020년 데이터
X_ = np.array(X_)
print(X_)

y_predicted_ = line_fitter.predict(X_)

plt.plot(y_predicted_, '.',c='red') #X축은 행 번호
#plt.plot(X,y_predicted_,c='blue')
plt.show()

