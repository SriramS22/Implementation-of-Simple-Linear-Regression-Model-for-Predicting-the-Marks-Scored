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
![Screenshot 2023-08-24 083338](https://github.com/SriramS22/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119094390/8c7d2a57-de46-4237-a542-26d2700ca9a3)

![Screenshot 2023-08-24 083348](https://github.com/SriramS22/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119094390/e17e9dcd-3881-4a6c-80ec-bd7552cd466c)

![Screenshot 2023-08-24 083355](https://github.com/SriramS22/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119094390/1f8e3ddf-a2b4-4169-8abe-df9ce11d435f)

![Screenshot 2023-08-24 083406](https://github.com/SriramS22/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119094390/be967a88-2bbc-4f5f-8535-e63b2a85ae4d)

![Screenshot 2023-08-24 083414](https://github.com/SriramS22/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119094390/5ecbfe61-c199-4077-a855-61c51f0256d5)

![Screenshot 2023-08-24 083422](https://github.com/SriramS22/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119094390/cc14f764-c4b7-49e4-b60f-5ad21441e3c2)

![Screenshot 2023-08-24 083429](https://github.com/SriramS22/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119094390/8ed1a10a-14ed-4805-a8cd-04b16380fd45)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
