from LogisticRegression import LogisticRegression
import numpy as np


#Testing Sigmoid function
x = 100
print(LogisticRegression.sigmoid(x))
x = -100
print(LogisticRegression.sigmoid(x))
x = np.array([i for i in range(10)])
print(list(map(LogisticRegression.sigmoid, x)))
x = np.array([i for i in range(-10,10)]).reshape(20,1)
print(x)
print(np.where(LogisticRegression.sigmoid(x) >= .5, 1, 0))

x = np.array([50,-50])
z = np.array((2*x + 5))

y = np.where(LogisticRegression.sigmoid(z) >= 0.5, 1, 0).reshape(len(x), 1)

myLogReg = LogisticRegression()
myLogReg.logistic_reg_fit(x, y)
print(y)
print("my coeff")
myLogReg.get_coeff()
myLogReg.predict(np.array([50, -50]))


# Testing single variable logistic regression
x = np.array([i for i in range(1, 16)])
y = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

myLogReg = LogisticRegression()
myLogReg.logistic_reg_fit(x, y)
print("my coeff")
myLogReg.get_coeff()
myLogReg.predict(x)


#Testing multivariable logistic regression
x = np.array([[1, 3], [2, 3], [3, 3], [3, 2], [4, 4], [1, 1], [1, 2], [2, 1]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1])

myLogReg = LogisticRegression()
myLogReg.logistic_reg_fit(x, y)
print("my coeff")
myLogReg.get_coeff()
myLogReg.predict(x)
myLogReg.predict(np.array([[2, 3], [1, 0.5], [1, 5]]))