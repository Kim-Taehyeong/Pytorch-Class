import sys
import torch
import torch.nn as nn
#               0    1    2     3   4
input_chars = ['h', 'i', 'e', 'l', 'o']

input_index = [0, 1, 0, 2, 3, 3]   # hihell
one_hot_encode = [[1, 0, 0, 0, 0],  # 0 h
                  [0, 1, 0, 0, 0],  # 1 i
                  [0, 0, 1, 0, 0],  # 2 e
                  [0, 0, 0, 1, 0],  # 3 l
                  [0, 0, 0, 0, 1]]  # 4 o

output_index = [1, 0, 2, 3, 3, 4]    # ihello
input_data = [one_hot_encode[x] for x in input_index]

input_data = torch.tensor(input_data)
tar_index = torch.tensor(output_index)


num_chars = 5
input_size = 5
hidden_size = 5
batch_size = 1
sequence_length = 1
num_layers = 1


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size, batch_first=True)

    def forward(self, cell, hidden, x):
        x = x.view(batch_size, sequence_length, input_size)
        # Input: (batch, seq_len, input_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        out, (hidden, cell) = self.rnn(x, (hidden, cell))
        return cell, hidden, out.view(-1, num_chars)

# Instantiate RNN model
model = Model()

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    loss = 0
    # Initial hidden values
    hidden = torch.zeros(num_layers, batch_size, hidden_size)
    cell = torch.zeros(num_layers, batch_size, hidden_size)

    sys.stdout.write("predicted string: ")
    for input, t_index in zip(input_data, tar_index):
        cell, hidden, output = model(cell, hidden, input.float())
        val, idx = output.max(1)
        sys.stdout.write(input_chars[idx.data[0]])
        loss += loss_func(output, torch.LongTensor([t_index]))

    print(", epoch: %d, loss: %1.3f" % (epoch + 1, loss))

    loss.backward()   # Compute Gradients
    optimizer.step()  # Update parameters

print("Learning finished!")