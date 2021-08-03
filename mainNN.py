import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

x = np.array([1,2])
weights = np.array([0.5,1])
oneNeuron = Neuron(weights,1)
print('Output of Neuron', oneNeuron.feedforward(x))


class exNeuralNetwork:
    def __init__(self):
        weight1 = np.array([0.5,1])
        weight2 = np.array([-0.5,0.5])
        weight3 = np.array([1,1])
        bias1 = 1
        bias2 = 2
        bias3 = 0

        self.h1 = Neuron(weight1,bias1)
        self.h2 = Neuron(weight2,bias2)
        self.o1 = Neuron(weight3,bias3)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1

x = np.array([1,2])
nNetwork = exNeuralNetwork()

print('Output of Neural Network:',nNetwork.feedforward(x))