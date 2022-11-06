# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe
2. Write a function computeCost to generate the cost function
3. Perform iterations og gradient steps with learning rate
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Varsha Ajith
RegisterNumber:  212221230118
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("student_scores.csv")
data.head()
data.isnull().sum()
x = data.Hours
x.head()
y = data.Scores
y.head()
n = len(x)
m = 0
c = 0
L = 0.001
loss = []
for i in range(10000):
    ypred = m*x + c
    MSE = (1/n) * sum((ypred - y)*2)
    dm = (2/n) * sum(x*(ypred-y))
    dc = (2/n) * sum(ypred-y)
    c = c-L*dc
    m = m-L*dm
    loss.append(MSE)
    #print(m)
print(m,c)
y_pred = m*x + c
plt.scatter(x,y,color = "pink")
plt.plot(x,y_pred)
plt.xlabel("Study hours")
plt.ylabel("Scores")
plt.title("Study hours vs. Scores")
plt.plot(loss)
plt.xlabel("Iterations")
plt.ylabel("loss") 

```
## Output:

![ml1](https://user-images.githubusercontent.com/94222288/200175017-ab7e11da-aaa7-4782-b446-87ebd4039783.png)
![ml2](https://user-images.githubusercontent.com/94222288/200175028-80d74ae4-f832-4533-b713-71f127e7ab1b.png)
![ml3](https://user-images.githubusercontent.com/94222288/200175030-81f51e78-4c1a-4510-a944-7e2a19a6eb03.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
