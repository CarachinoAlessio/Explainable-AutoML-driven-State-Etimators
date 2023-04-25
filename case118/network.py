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
