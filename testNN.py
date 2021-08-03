import numpy as np

x = np.array([1,2])
w1 = 0.5
w2 = 1
bias = 1

y = 1/(1+np.exp(-(w1*x[0]+w2*x[1]+bias)))

print('Output of Neuron:', y)