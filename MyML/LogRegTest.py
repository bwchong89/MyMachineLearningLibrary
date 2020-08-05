from LogisticRegression import LogisticRegression
import numpy as np

'''
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
'''
'''
x = np.array([50,-50])
z = np.array((2*x + 5))

y = np.where(LogisticRegression.sigmoid(z) >= 0.5, 1, 0).reshape(len(x), 1)

myLogReg = LogisticRegression()
myLogReg.logistic_reg_fit(x, y)
print(y)
print("my coeff")
myLogReg.get_coeff()
myLogReg.predict(np.array([50,-50]))
'''

#Testing single variable logistic regression
x = np.array([i for i in range(-30, 30)])
z = np.array((2*x + 5))

y = np.where(LogisticRegression.sigmoid(z) >= 0.5, 1, 0).reshape(len(x), 1)

myLogReg = LogisticRegression()
myLogReg.logistic_reg_fit(x, y)
print(y)
print("my coeff")
myLogReg.get_coeff()
myLogReg.predict(np.array(z))