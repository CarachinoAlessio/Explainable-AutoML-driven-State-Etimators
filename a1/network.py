import torch
import torch.nn as nn


class ANN(nn.Module):
    def init(self, n_input, n_hidden, n_output):
        super(ANN, self).init()
        self.hidden = nn.Linear(n_input, n_hidden)
        self.output = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # x = torch.sigmoid(self.hidden(x))
        x = nn.ReLU(self.hidden(x))
        x = self.output(x)
        return x
