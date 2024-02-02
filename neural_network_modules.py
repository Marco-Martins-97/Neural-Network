import numpy as np
np.random.seed(0)


class Dense_layer:
    def __init__(self, n_inputs, n_outouts):
        self.weights = np.random.randn(n_outouts, n_inputs)
        self.biases = np.random.randn(n_outouts, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.biases
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)
    
    def print(self):
        return self.weights, self.biases
    
class Tanh:
    def forward(self, input):
        self.input = input
        return np.tanh(self.input)
    
    def backward(self, output_gradient):
        return np.multiply(output_gradient, 1- np.tanh(self.input) ** 2)

