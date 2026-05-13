# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the dataset and split it into independent variables (features) and dependent variable (salary).
2. Divide the data into training and testing sets, then train the Decision Tree Regressor model using the training data.
3. Predict employee salary values using the trained Decision Tree Regressor on the test data.
4. Evaluate the model performance using regression metrics such as Mean Squared Error (MSE) and R² score.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: TEJA D P
RegisterNumber:212225040464  
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:/Users/acer/Downloads/Salary.csv")

le = LabelEncoder()
data['Position'] = le.fit_transform(data['Position'])

X = data.iloc[:, 1:2].values  
y = data.iloc[:, 2].values   

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])
print(f"Predicted Salary for Level 6.5: ${y_pred[0]}")

import numpy as np
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red', label='Actual Data')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Decision Tree Regression')
plt.title('Salary vs Level (Decision Tree Regressor)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()
```

## Output:
<img width="899" height="611" alt="image" src="https://github.com/user-attachments/assets/f095402d-fe2b-4074-b740-32fbac2e273e" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
