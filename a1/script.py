import torch
import torch.nn as nn
import scipy
import math
from torch.utils.data import DataLoader
import shap

import numpy as np
import parser
from a1.dataset import Dataset
from a1.network import ANN
from a1.train import train
from a1.test import test

args = parser.parse_arguments()
torch.manual_seed(42)
train_time = args.train
shap_values_time = args.shap_values
verbose = args.verbose
shapv2 = False

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

if train_time:
    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        # test(test_dataloader, model, loss_fn)
    print("Done!")
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

else:
    model.load_state_dict(torch.load("model.pth"))
    test(test_dataloader, model, loss_fn)

    if not shapv2:
        batch = next(iter(test_dataloader))
        measurements, _ = batch
        estimations = model(measurements).detach().numpy()
        background = measurements[:100].to(device)
        if shap_values_time:
            to_be_explained = measurements[100:150].to(device)

            explainer = shap.DeepExplainer(model, background)
            '''
            with open('explainer', 'wb') as f:
                explainer.save(f, model)
            with open('explainer', 'rb') as f:
                explainer = shap.DeepExplainer.load('explainer', model)
            '''

            shap_values = explainer.shap_values(to_be_explained)
            print("Saving shap_values on file...")
            with open('shap_values.npy', 'wb') as f:
                np.save(f, np.array(shap_values))
        else:
            print("Loading shap_values from file...")
            with open('shap_values.npy', 'rb') as f:
                shap_values = list(np.load(f))

            ## I am interested only in shap values, after 100 time instants, of voltage magnitudes of node #1 (50 time instants)
            shap_values_node_1 = shap_values[0]
            shap.summary_plot(shap_values_node_1, plot_type="bar", max_display=12, class_inds=[0])
            shap.summary_plot(shap_values_node_1, plot_type="violin", max_display=12, class_inds=[0])

            ## I am interested only in shap values of voltage magnitudes of node #1 (first time instant)
            node_1_estimation = estimations[100]
            #shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0])
            shap.plots._waterfall.waterfall_legacy(node_1_estimation[0], shap_values_node_1[0])


        # shap.plots.heatmap(shap_values)
        # shap.bar_plot(shap_values)
        # shap.plots.beeswarm(shap_values)
        # shap.plots.heatmap(shap_values, max_display=12)
        # shap.plots.bar(shap_values[0])

        # test(test_dataloader, model, loss_fn)
    else:
        batch = next(iter(test_dataloader))
        measurements, _ = batch
        background = measurements[:100].to(device)
        if shap_values_time:
            # to_be_explained = measurements[10:11].to(device)

            explainer = shap.KernelExplainer(model(background), data=background, link="identity")
            shap_values = explainer.shap_values(X=background[0:1, :])
            print("Saving shap_values on file...")

            with open('shap_valuesv2.npy', 'wb') as f:
                np.save(f, np.array(shap_values))
        else:
            print("Loading shap_values from file...")
            with open('shap_valuesv2.npy', 'rb') as f:
                shap_values = list(np.load(f))
