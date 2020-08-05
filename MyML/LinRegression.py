import numpy as np


class LinReg:

    def __init__(self):
        self.w = []
        self.b = 0
        self.loss = []

    def closed_form_fit(self, x, y):
        '''
        This assumes an X of size (m,n+1) & is invertible , w of size (n+1,1), and y of size (m,1)

        x = [[1 x1 x2 x3...xn]
            [1 x1 x2 x3...xn]
            [1 x1 x2 x3...xn]]

        w = [1 w1 w2 w3...wn]'

        y = [y1 y2 y3... yn]'

        w=(X' * X)^-1 * Y' * X
        '''

        print("Closed form Linear Regression")
        x = np.c_[np.ones(len(x)), x]
        w = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
        self.b = w[0]
        self.w = w[1:]

    def grad_descent_fit(self, x, y, alpha=0.00001, iterations=100000, regularization="none", lambd=0):
        print("Linear Regression Using Gradient Descent")
        n = x.ndim
        m = len(x)
        self.w = np.zeros((n, 1))

        for i in range(iterations):
            y_hat = np.dot(x.reshape(m, n), self.w) + self.b
            error = y_hat - y.reshape(m, 1)
            dw = (1 / m) * np.dot(x.T, error)
            db = (1 / m) * np.sum(error)
            if regularization == "none":
                self.w -= np.multiply(alpha, dw)
                self.b -= np.multiply(alpha, db)
            elif regularization == "L2":
                self.w -= np.multiply(alpha, dw) + lambd/m * np.multiply(self.w, self.w)
                self.b -= np.multiply(alpha, db)
            else:
                print("Please choose either L2 or None for regularization")

    def predict(self, x):
        return np.dot(x, self.w) + self.b

    def get_coeff(self):
        print("b = {}, w = {}".format(self.b, self.w))
