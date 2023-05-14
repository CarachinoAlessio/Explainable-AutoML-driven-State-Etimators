import torch
import torch.nn as nn
import scipy
import math
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import parser
from case118.dataset import Dataset
from case118.network import ANN
from case118.train import train
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
import pandas as pd
from tabulate import tabulate

args = parser.parse_arguments()
torch.manual_seed(42)
train_time = args.train
shap_values_time = args.shap_values
verbose = args.verbose

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def test(dataloader, model, loss_fn, plot_predictions=False):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            print(len(y[0]))
            for j in range(len(y[0])):
                if plot_predictions and batch == 0:
                    plt.figure(figsize=(10,6))
                    plt.plot(y[:, j])
                    plt.plot(pred[:, j])
                    plt.xlabel('t')
                    plt.ylabel('Voltage')
                    plt.title(f'Voltage on node #{j} ')
                    plt.legend(['Actual', 'Estimated'])
                    plt.show()
    test_loss /= num_batches
    mae = MeanAbsoluteError()
    mape = MeanAbsolutePercentageError()
    mse = MeanSquaredError()
    columns = ["MAE", "MAPE", "RMSE"]
    vmagnitudes_results = [[mae(pred, y), mape(pred, y),
                            torch.sqrt(mse(pred, y))]]

    print("------------------------------------------")
    print(f"     V Magnitudes")
    dt1 = pd.DataFrame(vmagnitudes_results, columns=columns)
    print(tabulate(dt1, headers='keys', tablefmt='psql', showindex=False))




data_x = np.load('../nets/net_18_data/data_x.npy')
data_y = np.load('../nets/net_18_data/data_y.npy')


if verbose:
    print(data_x.shape)
    print(data_y.shape)

# separate them into training 60%, test 40%
split_train = int(0.6 * data_x.shape[0])
test_x = data_x[:split_train, :]
test_y = data_y[:split_train, :]
train_x = data_x[split_train:, :]
train_y = data_y[split_train:, :]

if verbose:
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

train_data = Dataset(train_x, train_y)
train_dataloader = DataLoader(train_data, batch_size=100, drop_last=True)

test_data = Dataset(test_x, test_y)
# test_dataloader = DataLoader(test_data, batch_size=100, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=len(test_data))

# Train the model
input_shape = train_x.shape[1]
num_classes = train_y.shape[1]

model = ANN(input_shape, 2500, num_classes).to(device)
if verbose:
    print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if train_time and False:
    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        # test(test_dataloader, model, loss_fn)
    print("Done!")
    torch.save(model.state_dict(), "model_net18.pth")
    print("Saved PyTorch Model State to model.pth")

else:
    model.load_state_dict(torch.load("model_net18.pth"))
    test(test_dataloader, model, loss_fn, plot_predictions=True)
