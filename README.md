# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report b importing the required modules from sklearn.

6.Obtain the graph.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:  SHARANGINI T K
RegisterNumber:  212222230143

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]
X[:5]
y[:5]
plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()
def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)
x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)
def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)
def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)
def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)
np.mean(predict(res.x,X)==y)
```
## Output:
# Array value of X:
![image](https://github.com/shara56/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497104/af9e61da-025c-4e38-ab93-805585030aa8)
# Array value of Y:
![image](https://github.com/shara56/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497104/f30de9e1-e1b6-4705-9d80-4d40f7452eb1)
# Exam 1-Score graph:
![image](https://github.com/shara56/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497104/9ec57521-4c0c-4ed1-8cec-4387b7f0b18a)
# Sigmoid function graph:
![image](https://github.com/shara56/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497104/ba4b89bc-df99-4961-a2cc-c0b3f356b888)
# X_Train_grad value:
![image](https://github.com/shara56/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497104/ed807521-b815-4e20-969d-f1dc5647ef15)
# Y_Train_grad value:
![image](https://github.com/shara56/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497104/7a81a652-f8c1-475b-8db0-09726d8cbe8e)
# Print res.X:
![image](https://github.com/shara56/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497104/5cf61e9e-2465-41f3-89a0-84b4379d0de2)
# Decision boundary-gragh for exam score:
![image](https://github.com/shara56/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497104/fcf48242-2cb7-43c5-8061-3e77671778e7)
# Probability value:
![image](https://github.com/shara56/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497104/1d340c33-6400-42f4-ae6a-6d74722aaf81)
# Prediction value of mean:
![image](https://github.com/shara56/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497104/df0dfc8a-a94b-433f-8976-d4ef8ca0ff06)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

