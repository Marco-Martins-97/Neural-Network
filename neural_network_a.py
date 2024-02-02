import numpy as np
import neural_network_modules as nm
#XOR Neural Network

class Neural_Network:
    def __init__(self):
        self.learning_rate = 0.1
        self.error = 0
        self.loss = 0

        self.X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
        self.Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
        
        self.output = []
        self.output1 = []
        self.output2 = []
        self.output3 = []
        self.output4 = []
        self.gradient = [[0]]
        self.gradient1 = [[0]]
        self.gradient2 = [[0]]
        self.gradient3 = [[0]]
        self.gradient4 = [[0]]
    
    def print(self):
        return self.output, self.output1, self.output2, self.output3, self.output4, self.error, self.gradient, self.gradient1, self.gradient2, self.gradient3, self.gradient4
    '''
    network = [
       nm.Dense_layer(2, 3),
       nm.Tanh(),
       nm.Dense_layer(3, 1),
       nm.Tanh(),

    ] 
    '''
    dense1 = nm.Dense_layer(2, 3)
    tanh1 = nm.Tanh()
    dense2 = nm.Dense_layer(3, 1)
    tanh2 = nm.Tanh()
    
    def train(self):
   
        for x, y in zip(self.X, self.Y):
            #forward
            output = x
            '''
            for layer in self.network:
                output = layer.forward(output)

            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
            '''
            output1 = self.dense1.forward(output)
            output2 = self.tanh1.forward(output1)
            output3 = self.dense2.forward(output2)
            output4 = self.tanh2.forward(output3)
            
            self.error = np.mean(np.power(y - output4, 2))
            #print(error)
            grad = 2 * (output4 - y) / np.size(y)

            grad1 = self.tanh2.backward(grad)
            grad2 = self.dense2.backward(grad1, self.learning_rate)
            grad3 = self.tanh1.backward(grad2)
            grad4 = self.dense1.backward(grad3, self.learning_rate)

            self.output = output
            self.output1 = output1
            self.output2 = output2
            self.output3 = output3
            self.output4 = output4

            self.gradient = grad
            self.gradient1 = grad1
            self.gradient2 = grad2
            self.gradient3 = grad3
            self.gradient4 = grad4
        self.error /= len(x)
        #print(self.error)



