import torch
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.dropout = nn.Dropout(0.8)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.relu = nn.ReLU()
        self.hidden = nn.Linear(n_input, n_hidden)
        #self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.output = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # x = torch.sigmoid(self.hidden(x))
        try:
            x = self.hidden(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.output(x)
            return x

        except:
            print('I DO NOT WANT TO BE HERE')
            x = x.type(torch.FloatTensor)

            x = nn.functional.relu(self.hidden(x))
            x = self.dropout(x)
            #x = nn.functional.relu(self.hidden2(x))
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
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, znorm=None):
        super(LSTMStateEstimation, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.znorm = znorm
        if znorm is not None:
            self.emp_mean_x = torch.tensor(znorm[0])
            self.emp_mean_y = torch.tensor(znorm[1])
            self.emp_std_x = torch.tensor(znorm[2])
            self.emp_std_y = torch.tensor(znorm[3])
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).double()
        self.fc = nn.Linear(hidden_size, output_size).double()
        #self.init_weights()

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.double).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.double).to(x.device)
        #x = x.float()

        if self.znorm is not None:
            x = (x - self.emp_mean_x) / (self.emp_std_x + 1e-5)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Keep only hidden states of last time step in the sequence
        # to delete: out = out.contiguous().view(-1, self.hidden_size)
        out = out[:, -1, :]

        # Decode the hidden state of the last time step using a linear layer
        out = self.fc(out)

        if self.znorm is not None:
            out = (out * (self.emp_std_y + 1e-6)) + self.emp_mean_y

        return out


class GRUStateEstimation(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, znorm=None):
        super(GRUStateEstimation, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.znorm = znorm

        self.lstm = nn.GRU(input_size, hidden_size, num_layers, batch_first=True).double()
        self.fc = nn.Linear(hidden_size, output_size).double()
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.double).to(x.device)
        # x = x.float()

        if self.znorm is not None:
            x = (x - self.emp_mean_x) / (self.emp_std_x + 1e-5)

        # Forward propagate LSTM
        out, _ = self.lstm(x, h0)

        # Keep only hidden states of last time step in the sequence
        # to delete: out = out.contiguous().view(-1, self.hidden_size)
        out = out[:, -1, :]

        # Decode the hidden state of the last time step using a linear layer
        out = self.fc(out)

        if self.znorm is not None:
            out = (out * (self.emp_std_y + 1e-6)) + self.emp_mean_y

        return out