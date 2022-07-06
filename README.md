# Task-1
GRIP : THE SPARKS FOUNDATION
Data Science and Business Analytics
Author : Madipelly Aashritha
Task 1 : Prediction Using Supervised ML
In this task we have to predict the percentage score of a student based on the number of hours studied.The task has two variables where the feature is the number of hours studied and the target value is the percentage score.This can be solved using simple linear regression.
#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
Reading data from remote url
url ="http://bit.ly/w-data"
data = pd.read_csv(url)
Exploring data
print(data.shape)
data.head()
(25, 2)
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
data.describe()
Hours	Scores
count	25.000000	25.000000
mean	5.012000	51.480000
std	2.525094	25.286887
min	1.100000	17.000000
25%	2.700000	30.000000
50%	4.800000	47.000000
75%	7.400000	75.000000
max	9.200000	95.000000
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 25 entries, 0 to 24
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Hours   25 non-null     float64
 1   Scores  25 non-null     int64  
dtypes: float64(1), int64(1)
memory usage: 528.0 bytes
data.plot(kind ='scatter',x='Hours',y='Scores');
plt.show()

data.corr(method = "pearson")
Hours	Scores
Hours	1.000000	0.976191
Scores	0.976191	1.000000
data.corr(method = "spearman")
Hours	Scores
Hours	1.000000	0.971891
Scores	0.971891	1.000000
Hours = data["Hours"]
Scores= data["Scores"]
sns.histplot(Hours)
<AxesSubplot:xlabel='Hours', ylabel='Count'>

sns.histplot(Scores)
<AxesSubplot:xlabel='Scores', ylabel='Count'>

Linear Regression
x = data.iloc[:, :-1].values
y = data.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 50)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
LinearRegression()
m=reg.coef_
c=reg.intercept_
line=m*x+c
plt.scatter(x,y,color ='red')
plt.plot(x,line);
plt.show()

y_prediction=reg.predict(x_test)
print(y_prediction)
[88.21139357 28.71845267 69.02012231 39.27365186 13.36543566]
actual_predicted=pd.Series({'Target':y_test,'predicted':y_pred})
actual_predicted
Target                                    [95, 30, 76, 35, 17]
predicted    [88.21139357388516, 28.718452665057836, 69.020...
dtype: object
sns.set_style("whitegrid")
sns.histplot(np.array(y_test-y_pred))
plt.show()

What would be the prediction score if a student studies for 9.25 hours/day ?
h=9.25
s=reg.predict([[h]])
print("If a student studies for {} hours per day he/she will score {} % in exam".format(h,s))
If a student studies for 9.25 hours per day he/she will score [91.56986604] % in exam
Model Evaluation
from sklearn import metrics
from sklearn.metrics import r2_score
print("Mean absolute error:",metrics.mean_absolute_error(y_test,y_pred))
print("R2 score:",r2_score(y_test,y_pred))
Mean absolute error: 4.5916495300630285
R2 score: 0.971014141329942
