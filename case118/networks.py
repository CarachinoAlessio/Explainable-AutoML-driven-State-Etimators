import torch
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.hidden = nn.Linear(n_input, n_hidden)
        self.output = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # x = torch.sigmoid(self.hidden(x))
        x = nn.functional.relu(self.hidden(x))
        x = self.output(x)
        return x

'''
class LSTMStateEstimation(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMStateEstimation, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
'''


class LSTMStateEstimation(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMStateEstimation, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).double()
        self.fc = nn.Linear(hidden_size, output_size).double()

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.double).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.double).to(x.device)
        #x = x.float()


        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Keep only hidden states of last time step in the sequence
        # to delete: out = out.contiguous().view(-1, self.hidden_size)
        out = out[:, -1, :]

        # Decode the hidden state of the last time step using a linear layer
        out = self.fc(out)

        return out