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
from case118.networks import ANN, LSTMStateEstimation
from nets import Net18Dataset_LSTM
from case118.train import train
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
import pandas as pd
from tabulate import tabulate

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


data_x = np.load('../nets/net_18_data/measured_data_x.npy').astype(np.double)
data_y = np.load('../nets/net_18_data/data_y.npy').astype(np.double)
data_x = data_x[:len(data_x)-1]
data_y = data_y[:len(data_y)-1]
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

'''
train_data = Dataset(train_x, train_y)
train_dataloader = DataLoader(train_data, batch_size=16, drop_last=False)

val_data = Dataset(val_x, val_y)
val_dataloader = DataLoader(val_data, batch_size=int(len(val_data) / s), drop_last=False)

test_data = Dataset(test_x, test_y)
# test_dataloader = DataLoader(test_data, batch_size=100, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=len(test_data))
'''
sequence_length = 5
train_data = Net18Dataset_LSTM.Net18Dataset_LSTM(train_x, train_y, sequence_length)
train_dataloader = DataLoader(train_data, batch_size=32, drop_last=True)

val_data = Net18Dataset_LSTM.Net18Dataset_LSTM(val_x, val_y, sequence_length)
val_dataloader = DataLoader(val_data, drop_last=False)
#val_dataloader = DataLoader(val_data, batch_size=int(len(val_data) / s), drop_last=False)


test_data = Net18Dataset_LSTM.Net18Dataset_LSTM(test_x, test_y, sequence_length)
#test_dataloader = DataLoader(test_data)
test_dataloader = DataLoader(test_data, batch_size=len(test_data))



# Train the model
input_shape = train_x.shape[1]
num_classes = train_y.shape[1]

model = LSTMStateEstimation(input_shape, 32, num_classes).to(device)

if verbose:
    print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if train_time or True:
    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        # test(test_dataloader, model, loss_fn)
    print("Done!")
    torch.save(model.state_dict(), "model_lstm_net18.pth")
    torch.save(optimizer.state_dict(), "optimizer_lstm.pth")
    print("Saved PyTorch Model State to model_lstm_net18.pth")
    test(test_dataloader, model, loss_fn, plot_predictions=True)


'''
elif shap_values_time or True:
    model.load_state_dict(torch.load("model_net18.pth"))
    test(test_dataloader, model, loss_fn, plot_predictions=False)

    batch = next(iter(test_dataloader))
    measurements, _ = batch
    estimations = model(measurements).detach().numpy()
    background = measurements[:100].to(device)
    if shap_values_time and False:
        to_be_explained = measurements[100:150].to(device)

        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(to_be_explained)
        print("Saving shap_values on file...")
        with open('shap_values_net18.npy', 'wb') as f:
            np.save(f, np.array(shap_values))
    else:
        print("Loading shap_values from file...")
        with open('shap_values_net18.npy', 'rb') as f:
            shap_values = list(np.load(f))

        ## I am interested only in shap values, after 100 time instants, of voltage magnitudes of node #1 (50 time instants)
        shap_values_node_1 = shap_values[5]
        shap.summary_plot(shap_values_node_1, plot_type="bar", max_display=12)
        shap.summary_plot(shap_values_node_1, plot_type="violin", max_display=12)

        ## I am interested only in shap values of voltage magnitudes of node #1 (first time instant)
        node_1_estimation = estimations[100]
        # shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0])
        shap.plots._waterfall.waterfall_legacy(node_1_estimation[0], shap_values_node_1[0])

        data = shap_values_node_1[0][:55].reshape((1, 55))
        data_indices = sorted(range(len(data.flatten())), key=lambda x: -abs(data.flatten()[x]))
        print(data_indices)
        plt.figure()
        fig, ax = plt.subplots(figsize=(100, 5))

        im = ax.imshow(data)

        ax.set_xticks(np.arange(len(data[0])))
        ax.set_yticks(np.arange(len(data)))
        ax.set_xticklabels(
            ['f' + str(i) for i in range(data.shape[1])])
        ax.set_yticklabels(['row1'])

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor", fontsize=72)

        ax.set_title("Node 1 voltage magnitude explanation at t=100")
        fig.tight_layout()
        plt.show()
        print("All plots have been generated.")

elif retrain_time:
    print("--------- Starting the retraining process ---------")
    incr_dataloader, shap_list = get_incr_data_and_shap_values(val_dataloader, model, loss_fn, voltage_threshold=0.07)
    #shap_multiplier = 1
    #shap_offset = 0.2
    shap_sum = np.sum(shap_list, axis=2).T  # result size: (56, 18)
    shap_sum = np.sum(shap_sum, axis=1)

    weights = torch.from_numpy(abs((shap_sum - np.mean(shap_sum)) / np.std(shap_sum)))

    epochs = 100
    loss_fn = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer.load_state_dict(torch.load('optimizer.pth'))
    for t in range(epochs):
        retrain(incr_dataloader, weights, model, loss_fn, optimizer)
    print("Done!")
    torch.save(model.state_dict(), "retrained_model.pth")
    print("Saved PyTorch Model State to retrained_model.pth")

test_retrained = False
if test_retrained:
    model.load_state_dict(torch.load("retrained_model.pth"))
    print("--------- Testing the retrained model... ---------")
    baseline_model = ANN(input_shape, 2500, num_classes).to(device)
    baseline_model.load_state_dict(torch.load("model_net18.pth"))
    loss_fn = nn.MSELoss()
    test_after_retrain(test_dataloader, model, baseline_model, loss_fn, plot_predictions=True)
'''
print("--------- Done ---------")



