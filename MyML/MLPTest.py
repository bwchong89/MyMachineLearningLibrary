import numpy as np
from MLP import NeuralNetwork

#Test activations
x = np.array([i for i in range(-4, 5)])
print("Sigmoid test:\n{}".format(NeuralNetwork.sigmoid(x)))
print("Tanh test:\n{}".format(NeuralNetwork.tanh(x)))
print("ReLu test:\n{}".format(NeuralNetwork.relu(x)))

#Test initilization
TestSize = [2, 3, 4]
activation = ["sigmoid","sigmoid"]
myNN = NeuralNetwork(TestSize, activation)
myNN.initializeNN()
myNN.show_parameters()

#Test forward prop
x = np.array([[1,2],[3, 4],[5,6]])
myNN.NN_fit(x, 2)

#Test backprop

