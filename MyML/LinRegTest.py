from LinRegression import LinReg
import numpy as np

#test 1:
#y = 2x+3
#expected output: b = 3, w = 2

x1 = np.array([i for i in range(100)]).T
y1 = np.array([2*i+3 for i in range(100)]).T

#test 2:
#y = 2x1+3x2+4
#expected output: b = 3, w = 2

x2 = np.array([[1, 1], [1,2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]])
y2 = np.array([9, 12, 15, 11, 14, 17, 13, 16, 19]).T

#Closed form
#single linear regression
mylinreg = LinReg()
mylinreg.closed_form_fit(x1, y1)
print("Expected output: b = 3, w = 2 \nOutput:")
mylinreg.get_coeff()
print("Expected output: 43 \nOutput: {}\n".format(mylinreg.predict(20)))

#Multiple linear regression
mylinreg.closed_form_fit(x2, y2)
print("Expected output: b = 4, w = [2, 3] \nOutput: ")
mylinreg.get_coeff()
print("Expected output: 28, 24 \nOutput: {}\n".format(mylinreg.predict([[3, 6],[7, 2]])))

#grad form
#single linear regression
mylinreg.grad_descent_fit(x1, y1)
print("Expected output: b = 3, w = 2 \nOutput:")
mylinreg.get_coeff()
print("Expected output: 43 \nOutput: {}\n".format(mylinreg.predict(20)))

#Multiple linear regression
mylinreg.grad_descent_fit(x2, y2)
print("Expected output: b = 4, w = [2, 3] \nOutput: ")
mylinreg.get_coeff()
print("Expected output: 28, 24 \nOutput: {}\n".format(mylinreg.predict([[3, 6],[7, 2]])))

#Multiple linear regression with regularization
mylinreg.grad_descent_fit(x2, y2, regularization="L2", lambd=0.001)
print("Expected output: b = 4, w = [2, 3] \nOutput: ")
mylinreg.get_coeff()
print("Expected output: 28, 24 \nOutput: {}\n".format(mylinreg.predict([[3, 6],[7, 2]])))
