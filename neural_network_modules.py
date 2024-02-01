import numpy as np
np.random.seed(10)


class Dense_layer:
    def __init__(self, n_inputs, n_outouts):
        self.weights = np.random.randn(n_outouts, n_inputs)
        self.biases = np.random.randn(n_outouts, 1)
    
    def print(self):
        return self.weights, self.biases