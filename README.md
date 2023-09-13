# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sriram S 
RegisterNumber:  22009336
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: A.Anbuselvam
RegisterNumber:  22009081
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
## Head

![Screenshot 2023-09-12 153046](https://github.com/SriramS22/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119094390/3e9c6790-d98d-4875-9fe8-bca8fbcbbcc7)

## Tail

![Screenshot 2023-09-12 153120](https://github.com/SriramS22/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119094390/9c575f6b-ac0f-4fe8-add1-a14122e0219e)

## Array values of x

![Screenshot 2023-09-12 153157](https://github.com/SriramS22/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119094390/f5feb380-c1c5-496f-bfd5-275b7ad25a72)

## Array values of y

![Screenshot 2023-09-12 153226](https://github.com/SriramS22/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119094390/c74a9af1-cc77-4938-8423-08274ea90cf8)

## Values of y prediction

![Screenshot 2023-09-12 153307](https://github.com/SriramS22/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119094390/b8003b46-5f19-4068-92ff-b0f99707d0dc)

## Values of y test

![Screenshot 2023-09-12 153342](https://github.com/SriramS22/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119094390/10df3f2c-7d8a-4a8d-ad42-7f7341766659)

## Training set graph

![Screenshot 2023-09-12 153420](https://github.com/SriramS22/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119094390/645d8f0c-6229-4662-ad0c-0942416a01b2)

## Test set graph

![Screenshot 2023-09-12 153456](https://github.com/SriramS22/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119094390/0177519d-11eb-4080-830e-9d04c3311ee2)

## Values of MSE,MAE,RMSE

![Screenshot 2023-09-12 153521](https://github.com/SriramS22/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119094390/1ff6ada1-9f2b-42cf-8b1c-35298a2aabec)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
