import torch
import torch.nn as nn

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define input size, hidden layer size, and output size
n_input = 784
n_hidden = 256
n_output = 10


# Create custom class for neural network
class NeuralNetwork(nn.Module):
    def init(self):
        super(NeuralNetwork, self).init()
        self.hidden = nn.Linear(n_input, n_hidden)
        self.output = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = self.output(x)
        return x


model = NeuralNetwork()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train neural network
for epoch in range(100):
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
