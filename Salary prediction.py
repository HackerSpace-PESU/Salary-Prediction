# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:34:18 2020

@author: samud
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_data.csv')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 10)
X_poly = poly_reg.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg2 = LinearRegression()
lin_reg2.fit(X,y)

#visualising polynomial regression
X_grid = np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape((len(X_grid)),1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#polynomial regression for smooth curve
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#Linear regression
plt.scatter(X, y, color = 'red')
plt.title('Linear Regression')
plt.xlabel('Experience')
plt.ylabel('Salary')
model=LinearRegression()
model.fit(X,y)
y_pred=model.predict(X)
plt.plot(X,y_pred,color='blue')
plt.show()

import tkinter as tk
root= tk.Tk()
root.title("Enter Years of experience")

e=tk.Entry(root,width=20)
e.pack()

def myClick():
    n=e.get()
    sal=lin_reg.predict(poly_reg.fit_transform([[n]]))
    sal2=model.coef_[0]*float(n)
    myLabel=tk.Label(root,text="Salary from polynomial regression is "+str(sal)+ " and Salary from linear regression is "+str(sal2)) 
    myLabel.pack()
    
myButton=tk.Button(root,text="Enter",command=myClick)
myButton.pack()
root.mainloop()