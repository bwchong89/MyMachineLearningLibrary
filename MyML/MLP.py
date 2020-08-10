import numpy as np

class NeuralNetwork:

    def __init__(self, size, activation):
        self.parameters = {}
        self.size = size
        self.activation = activation
        self.cost = []

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def tanh(x):
        return np.tanh(x)

    def relu(x):
        return np.maximum(0,x)

    def initializeNN(self):
        L = len(self.size)

        for i in range(L-1):
            self.parameters["W" + str(i+1)] = np.random.randn(self.size[i], self.size[i+1])
            self.parameters["b" + str(i+1)] = 0

    def show_parameters(self):
        print("Parameters:")
        print(self.parameters)

    def forward_prop(self, x):
        A = [x]
        L = len(self.size)
        print("A0: {}".format(A[0]))
        print("w0: {}".format(self.parameters["W" + str(1)]))
        print("b0: {}".format(self.parameters["b" + str(1)]))
        for i in range(L-1):
            z = np.dot(A[i], self.parameters["W"+str(i+1)]) + self.parameters["b"+str(i+1)]

            if self.activation[i] == "sigmoid":
                print("activation {} is a {}".format(i+1, self.activation[i]))
                A.append(NeuralNetwork.sigmoid(z))
            elif self.activation[i] == "tanh":
                print("activation {} is a {}".format(i+1, self.activation[i]))
                A.append(NeuralNetwork.tanh(z))
            elif self.activation[i] == "relu":
                print("activation {} is a {}".format(i+1, self.activation[i]))
                A.append(NeuralNetwork.tanh(z))
            else:
                print("incorrect activation function")
            print(A[i+1])

        print("final activation is a sigmoid")
        z = np.dot(A[i], self.parameters["W" + str(L-1)]) + self.parameters["b" + str(L-1)]
        np.array(A.append(NeuralNetwork.sigmoid(z)))
        A = np.where(A > 0.5, 1, 0)
        print("final")
        print(A[L])
        return A

    def backward_prop(self, y):
        print(y)

    def NN_fit(self, x, y, iterations=10000, alpha=0.001):
        NeuralNetwork.forward_prop(self, x)
        NeuralNetwork.backward_prop(self, y)







