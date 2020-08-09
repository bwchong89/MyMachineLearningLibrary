import numpy as np

class NeuralNetwork:

    def __init__(self,size,activation):
        self.parameters = {}
        self.size = size
        self.activation = activation
        self.cost = []

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def tanh(x):
        return np.tanh(x)

    def relu(x):
        return max(0, x)

    def initializeNN(self):
        pass