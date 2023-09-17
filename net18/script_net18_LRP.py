import torch
import torch.nn as nn
import scipy
import math

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import shap
import numpy as np
import matplotlib.pyplot as plt
import parser
from case118.dataset import Dataset
from case118.networks import ANN
from case118.train import train
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
import pandas as pd
from tabulate import tabulate
from zennit.composites import EpsilonGammaBox
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient


args = parser.parse_arguments()
torch.manual_seed(42)
train_time = args.train
retrain_time = args.retrain_time
test_retrain = args.test_retrained
s = args.s
shap_values_time = False

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
                    plt.legend(['Real', 'Estimated'])
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


data_x = np.load('../nets/net_18_data/measured_data_x.npy')
data_y = np.load('../nets/net_18_data/data_y.npy')
#measured_x = np.load('../nets/net_18_data/measured_data_x.npy')
#measured_y = np.load('../nets/net_18_data/measured_data_y.npy')

if verbose:
    print(data_x.shape)
    print(data_y.shape)

# separate them into training 60%, test 40%
split_train = int(0.6 * data_x.shape[0])
train_x = data_x[:split_train, :]
train_y = data_y[:split_train, :]

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3, shuffle=False)

test_x = data_x[split_train:, :]
test_y = data_y[split_train:, :]


if verbose:
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    print(val_x.shape)
    print(val_y.shape)

train_data = Dataset(train_x, train_y)
train_dataloader = DataLoader(train_data, batch_size=16, drop_last=False)

val_data = Dataset(val_x, val_y)
val_dataloader = DataLoader(val_data, batch_size=int(len(val_data) / s), drop_last=False)

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

'''
if train_time or True:
    epochs = 30
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
    print("Done!")
    torch.save(model.state_dict(), "model_net18.pth")
    torch.save(optimizer.state_dict(), "optimizer.pth")
    print("Saved PyTorch Model State to model.pth")
    test(test_dataloader, model, loss_fn, plot_predictions=True)
'''
model.load_state_dict(torch.load("model_net18.pth"))
model.eval()

canonizers = [SequentialMergeBatchNorm()]
composite = EpsilonGammaBox(low=-3., high=3., canonizers=canonizers)

(X, Y) = next(iter(test_dataloader))
X = X.to(device)

with Gradient(model=model, composite=composite) as attributor:
    #plt.plot(X[:,0])
    #plt.show()
    out, relevance = attributor(X[200], torch.eye(18)[[0]].squeeze())

relevance = relevance.numpy()
relevance = relevance + abs(min(relevance.ravel()))
norm_relevance = ((relevance - min(relevance)) / (max(relevance) - min(relevance)))

print(relevance)
plt.imshow(norm_relevance.reshape((5, 11)))
plt.colorbar()

plt.show()

print("--------- Done ---------")



