import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  return ((y_true - y_pred) ** 2).mean()

class exNeuralNetwork:
  def __init__(self):
    # Weights
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    self.x_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
    self.h1 = sigmoid(self.x_h1)

    self.x_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
    self.h2 = sigmoid(self.x_h2)

    self.x_o1 = self.w5 * self.h1 + self.w6 * self.h2 + self.b3
    self.o1 = sigmoid(self.x_o1)
    return self.o1

  def gradient(self, x, y_true):

    self.do1_w1 = -2 * (y_true - self.o1)*d_sigmoid(self.x_o1)*self.w5*d_sigmoid(self.x_h1)*x[0]
    self.do1_w2 = -2 * (y_true - self.o1)*d_sigmoid(self.x_o1)*self.w5*d_sigmoid(self.x_h1)*x[1]
    self.do1_b1 = -2 * (y_true - self.o1)*d_sigmoid(self.x_o1)*self.w5*d_sigmoid(self.x_h1)*1

    self.do1_w3 = -2 * (y_true - self.o1)*d_sigmoid(self.x_o1)*self.w6*d_sigmoid(self.x_h2)*x[0]
    self.do1_w4 = -2 * (y_true - self.o1)*d_sigmoid(self.x_o1)*self.w6*d_sigmoid(self.x_h2)*x[1]
    self.do1_b2 = -2 * (y_true - self.o1)*d_sigmoid(self.x_o1)*self.w6*d_sigmoid(self.x_h2)*1

    self.do1_w5 = -2 * (y_true - self.o1)*d_sigmoid(self.x_o1)*self.h1
    self.do1_w6 = -2 * (y_true - self.o1)*d_sigmoid(self.x_o1)*self.h2
    self.do1_b3 = -2 * (y_true - self.o1)*d_sigmoid(self.x_o1)*1
    return

  def train(self, data, all_y_trues):
    learn_rate = 0.05
    epochs = 1000 # number of times to loop through the entire dataset
    mse_pl = []

    for x in data:
      print('Prediction before training:', self.feedforward(x))

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        self.feedforward(x)

        # Compute Gradients
        self.gradient(x, y_true)


        # --- Update weights and biases with Stochastic Gradient Descent Algorithm
        # Neuron h1
        self.w1 -= learn_rate * self.do1_w1
        self.w2 -= learn_rate * self.do1_w2
        self.b1 -= learn_rate * self.do1_b1

        # Neuron h2
        self.w3 -= learn_rate * self.do1_w3
        self.w4 -= learn_rate * self.do1_w4
        self.b2 -= learn_rate * self.do1_b2

        # Neuron o1
        self.w5 -= learn_rate * self.do1_w5
        self.w6 -= learn_rate * self.do1_w6
        self.b3 -= learn_rate * self.do1_b3

      # Compute MSE in each epoch
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))
        mse_pl.append(loss)

    for x in data:
      print('Prediction after training:', self.feedforward(x))
    plt.plot(mse_pl)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.show()


# Define dataset
data = np.array([
  [-4, -3],  # 여자 1의 몸무게 키 및 몸무게의 평균에서의 차이값
  [27, 8],   # 남자 1
  [12, 8],   # 남자 2
  [-19, -9], # 여자 2
])
all_y_trues = np.array([
  1, # 여자
  0, # 남자
  0, # 남자
  1, # 여자
])

# Initialize Network
network = exNeuralNetwork()

# Train Network
network.train(data, all_y_trues)