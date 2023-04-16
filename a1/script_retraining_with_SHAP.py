import torch
import torch.nn as nn
import scipy
import math
from torch.utils.data import DataLoader

import numpy as np
import parser
from a1.dataset import Dataset
from a1.network import ANN
from a1.retrain import get_incr_data_and_shap_values, retrain
from a1.train import train
from a1.test import test_before_retraining
from sklearn.model_selection import train_test_split

args = parser.parse_arguments()
torch.manual_seed(42)
train_time = args.train
retrain_time = False
shap_values_time = args.shap_values
verbose = args.verbose
s = args.s

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# data loading part
caseNo = 118
weight_4_mag = 100
weight_4_ang = 180 / math.pi

data = scipy.io.loadmat('data_for_SE_case' + str(caseNo) + '_for_ML.mat')
# print(data['inputs'].shape, data['labels'].shape)

data_x = data['Z_measure']
data_y = data['Output']

# this is because measurement are available for 4 and half years but state estimation is available only for two years for IEEE 118
data_x = data_x[0:data_y.shape[0], :]

data_y[:, 0:caseNo] = weight_4_mag * data_y[:, 0:caseNo]
data_y[:, caseNo:] = weight_4_ang * data_y[:, caseNo:]
if verbose:
    print(data_x.shape)
    print(data_y.shape)

# separate them into training 60%, test 40%
split_train = int(0.4 * data_x.shape[0])
test_x = data_x[:split_train, :]
test_y = data_y[:split_train, :]
train_x = data_x[split_train:, :]
train_y = data_y[split_train:, :]
# separate them into training 80%, val 20%
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, shuffle=False)

if verbose:
    print(train_x.shape)
    print(train_y.shape)
    print(val_x.shape)
    print(val_y.shape)
    print(test_x.shape)
    print(test_y.shape)

train_data = Dataset(train_x, train_y)
train_dataloader = DataLoader(train_data, batch_size=100, drop_last=True)

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

if train_time:
    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        if t == 0 or t % 50 == 0:
            test_before_retraining(val_dataloader, model, loss_fn, plot_predictions=True)
    print("Done!")
    torch.save(model.state_dict(), "model_baseline.pth")
    print("Saved PyTorch Model State to model_baseline.pth")


else:
    model.load_state_dict(torch.load("model_baseline.pth"))
print("--------- Testing the baseline... ---------")
test_before_retraining(test_dataloader, model, loss_fn, no_samples=200, plot_predictions=True)

print("--------- Starting the retraining process ---------")
incr_dataloader, shap_list = get_incr_data_and_shap_values(val_dataloader, model, loss_fn, voltage_threshold=0.27)
shap_multiplier = 10
shap_sum = np.sum(shap_list[:118], axis=2).T  # result size: (58, 118)
shap_sum = np.sum(shap_sum, axis=1)

weights = torch.from_numpy(shap_multiplier * abs((shap_sum - np.mean(shap_sum)) / np.std(shap_sum)))

if retrain_time:
    epochs = 1000
    loss_fn = nn.MSELoss(reduction='none')
    for t in range(epochs):
        retrain(incr_dataloader, weights, model, loss_fn, optimizer)
    print("Done!")
    torch.save(model.state_dict(), "retrained_model.pth")
    print("Saved PyTorch Model State to retrained_model.pth")

model.load_state_dict(torch.load("retrained_model.pth"))
print("--------- Testing the baseline... ---------")
loss_fn = nn.MSELoss()
test_before_retraining(test_dataloader, model, loss_fn, no_samples=200, plot_predictions=True)
print("--------- Done ---------")
