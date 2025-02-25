import torch
import torch.nn as nn
import scipy
import math

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import shap

import numpy as np
import matplotlib.pyplot as plt
import parser
from case118.dataset import Dataset
from case118.networks import LSTMStateEstimation
from case118.train import train
from case118.test import test
from nets import Net18Dataset_LSTM

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

# separate them into training 40%, test 60%
split_train = int(0.6 * data_x.shape[0])
test_x = data_x[:split_train, :]
test_y = data_y[:split_train, :caseNo]
train_x = data_x[split_train:, :]
train_y = data_y[split_train:, :caseNo]

'''
m = train_x.mean(axis=0)
s = train_x.std(axis=0)

train_x = (train_x - train_x.mean(axis=0) )/ (train_x.std(axis=0) + 1e-6)

#train_x = (train_x * (s+ 1e-6)) + m

train_y = (train_y - train_y.mean(axis=0) )/ (train_y.std(axis=0) + 1e-6)
test_x = (test_x - test_x.mean(axis=0) )/ (test_x.std(axis=0) + 1e-6)
test_y = (test_y - test_y.mean(axis=0) )/ (test_y.std(axis=0) + 1e-6)
'''
# Train the model
input_shape = train_x.shape[1]
num_classes = train_y.shape[1]

#model = ANN(input_shape, 2500, num_classes).to(device)
model = LSTMStateEstimation(input_shape, 250, num_classes, num_layers=5, znorm=[train_x.mean(axis=0), train_y.mean(axis=0), train_x.std(axis=0), train_y.std(axis=0) ]).to(device)

if verbose:
    print(model)

if verbose:
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

sequence_length = 5
train_data = Net18Dataset_LSTM.Net18Dataset_LSTM(train_x, train_y, sequence_length)

#train_data = Dataset(train_x, train_y)
train_dataloader = DataLoader(train_data, batch_size=32, drop_last=True)

# test_data = Dataset(test_x, test_y)
# test_dataloader = DataLoader(test_data, batch_size=100, drop_last=True)
test_data = Net18Dataset_LSTM.Net18Dataset_LSTM(test_x, test_y, sequence_length)
test_dataloader = DataLoader(test_data, batch_size=len(test_data))





loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if train_time or True:
    epochs = 60
    for t in range(epochs):
        best_loss = 1000
        print(f"Epoch {t + 1}\n-------------------------------")
        l = train(train_dataloader, model, loss_fn, optimizer)
        if l < best_loss:
            best_loss = l
            torch.save(model.state_dict(), "model_lstm.pth")
            print("Saved PyTorch Model State to model.pth")
        test(test_dataloader, model, loss_fn, plot_predictions=True)
    print("Done!")


else:
    model.load_state_dict(torch.load("model_lstm.pth"))
    test(test_dataloader, model, loss_fn, plot_predictions=True)

    batch = next(iter(test_dataloader))
    measurements, _ = batch
    estimations = model(measurements).detach().numpy()
    background = measurements[:100].to(device)
    if shap_values_time:
        to_be_explained = measurements[100:150].to(device)

        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(to_be_explained)
        print("Saving shap_values on file...")
        with open('shap_values_lstm.npy', 'wb') as f:
            np.save(f, np.array(shap_values))
    else:
        print("Loading shap_values from file...")
        with open('shap_values_lstm.npy', 'rb') as f:
            shap_values = list(np.load(f))

        ## I am interested only in shap values, after 100 time instants, of voltage magnitudes of node #1 (50 time instants)
        shap_values_node_1 = shap_values[0]
        shap.summary_plot(shap_values_node_1, plot_type="bar", max_display=12, class_inds=[0])
        shap.summary_plot(shap_values_node_1, plot_type="violin", max_display=12, class_inds=[0])

        ## I am interested only in shap values of voltage magnitudes of node #1 (first time instant)
        node_1_estimation = estimations[100]
        # shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0])
        shap.plots._waterfall.waterfall_legacy(node_1_estimation[0], shap_values_node_1[0])

        data = shap_values_node_1[0][:562].reshape((1, 562))

        plt.figure()
        fig, ax = plt.subplots(figsize=(100, 5))

        im = ax.imshow(data)

        ax.set_xticks(np.arange(len(data[0])))
        ax.set_yticks(np.arange(len(data)))
        ax.set_xticklabels(
            ['f' + str(i) for i in range(data.shape[1])])
        ax.set_yticklabels(['row1'])

        plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
                 rotation_mode="anchor")

        ax.set_title("Node 1 voltage magnitude explanation at t=100")
        fig.tight_layout()
        plt.show()
        print("All plots have been generated.")

    # test(test_dataloader, model, loss_fn)
