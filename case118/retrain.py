import os
from torch.utils.data import DataLoader
import numpy as np
import parser
from case118.dataset import Dataset
import shap

args = parser.parse_arguments()
device = "cpu"
s = args.s  # this script currently works only with s=1
compute_shap_for_val = False if os.path.isfile('shap_values_val.npy') else True


def get_incr_data_and_shap_values(dataloader, model, loss_fn, voltage_threshold=0.3):
    num_batches = len(dataloader)
    model.eval()
    # with torch.no_grad():
    incr_data_x = np.empty((0, 562))  # 562 is the size of input measurements
    incr_data_y = np.empty((0, 236))  # 236 is the size of output measurements
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        # X = X.detach().numpy()
        # y = y.detach().numpy()
        pred = pred.detach().numpy()
        mae_per_row = abs(y[:, :128] - pred[:, :128]).mean(axis=1)
        above_threshold_index = mae_per_row >= voltage_threshold
        under_threshold_index = ~above_threshold_index

        incr_data_x_batch = X[above_threshold_index]
        print(f"{len(X[above_threshold_index])}")
        incr_data_y_batch = y[above_threshold_index]
        incr_data_x = np.vstack((incr_data_x, incr_data_x_batch))
        incr_data_y = np.vstack((incr_data_y, incr_data_y_batch))

        if compute_shap_for_val:
            explainer = shap.DeepExplainer(model, X[under_threshold_index][:100])
            shap_values = explainer.shap_values(X[above_threshold_index])
            print("Saving shap_values on file...")
            with open('shap_values_val.npy', 'wb') as f:
                np.save(f, np.array(shap_values))
        else:
            print("Loading shap_values from file...")
            with open('shap_values_val.npy', 'rb') as f:
                shap_values = list(np.load(f))
        break

    incr_data = Dataset(incr_data_x, incr_data_y)
    incr_dataloader = DataLoader(incr_data, batch_size=int(len(incr_data) / s), drop_last=False)
    return incr_dataloader, shap_values


def retrain(dataloader, weights, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    #weights = torch.ones(58)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y).T*weights
        loss = loss.T.mean()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if batch == (len(dataloader) - 1):
        #    loss, current = loss.item(), (batch + 1) * len(X)
        #    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
