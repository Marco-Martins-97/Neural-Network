import numpy as np
import neural_network_modules as nm
#Native Neural Network

class Neural_Network:
    def __init__(self):
        self.learning_rate = 0.1
        self.error = 0

        self.X = X
        self.Y = y

    self.network = [
        nm.Dense(2,3),
        nm.Tanh(),
        nm.Dense(3,1),
        nm.Tanh()
    ]
    
    def train(self):
   
        for x, y in zip(self.X, self.Y):
            #forward
            output = x
            for layer in self.network:
                output = layer.forward(output)
            
            self.error = nm.mse(y, output)
            #print(error)
            grad = nm.mse_derivative(y, output)


            for layer in reversed(self.network):
                grad = layer.backward(grad, self.learning_rate)

        self.error /= len(x)
        #print(self.error)

