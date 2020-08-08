import numpy as np

class LogisticRegression:

    def __init__(self):
        self.w = []
        self.b = 0
        self.cost = []

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def logistic_reg_fit(self, x, y, alpha=0.00001, iterations=10000, regularization="none", lambd=0):
        n = x.ndim
        m = len(x)
        self.w = np.zeros((n, 1))
        self.b = 0
        y = y.reshape(m, 1)

        for i in range(iterations):
            z = np.dot(x.reshape(m, n), self.w.reshape(n, 1)) + self.b
            y_hat = LogisticRegression.sigmoid(z).reshape(m, 1)
            y_hat = np.where(y_hat > 0.5, 1, 0)

            if i % (iterations // 10) == 0:
                cost = (1/m)*np.sum(-y*np.log(y_hat)-(1-y)*np.log(1-y_hat))
                self.cost.append(cost)

            dw = (1/m) * np.dot(x.reshape(m, n).T, (y_hat-y).reshape(m, 1))
            db = (1/m) * np.sum((y_hat-y))

            if regularization == "none":
                self.w -= alpha*dw
                self.b -= alpha*db
            elif regularization == "L2":
                self.w -= alpha*dw + lambd/m * np.multiply(self.w, self.w)
                self.b -= alpha * db
            else:
                print("Please choose either L2 or none for regularization")

        self.cost.append((1 / m) * np.sum(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)))


    def get_coeff(self):
        print("b = {}, w = {}".format(self.b, self.w))

    def predict(self, x):
        n = x.ndim
        m = len(x)
        z = np.dot(x.reshape(m, n), self.w.reshape(n, 1)) + self.b
        y_hat = LogisticRegression.sigmoid(z)
        print(np.where(y_hat > 0.5, 1, 0))

