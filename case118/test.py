import torch
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
import pandas as pd
from tabulate import tabulate

device = "cpu"
caseNo = 118


def test(dataloader, model, loss_fn, plot_predictions=False):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            print(f"Results for node 1 (voltage magnitude) after 100 time instants:\nGT: {y[100][0]}, ES: {pred[100][0]}")
            test_loss += loss_fn(pred, y).item()
            if plot_predictions and batch == 0:
                plt.figure(figsize=(10,6))
                plt.plot(np.arange(200), y[:200, 0])
                plt.plot(np.arange(200), pred[:200, 0])
                plt.xlabel('t')
                plt.ylabel('Voltage')
                plt.title('Voltage on node #1 (First 200 time instants)')
                plt.legend(['Actual', 'Estimated'])
                plt.show()
    test_loss /= num_batches
    mae = MeanAbsoluteError()
    mape = MeanAbsolutePercentageError()
    mse = MeanSquaredError()
    columns = ["MAE", "MAPE", "RMSE"]
    vmagnitudes_results = [[mae(pred[:, :caseNo], y[:, :caseNo]), mape(pred[:, :caseNo], y[:, :caseNo]),
                            torch.sqrt(mse(pred[:, :caseNo], y[:, :caseNo]))]]
    vangles_results = [[mae(pred[:, caseNo:], y[:, caseNo:]), None, torch.sqrt(mse(pred[:, caseNo:], y[:, caseNo:]))]]

    print("------------------------------------------")
    print(f"     V Magnitudes")
    dt1 = pd.DataFrame(vmagnitudes_results, columns=columns)
    print(tabulate(dt1, headers='keys', tablefmt='psql', showindex=False))

    print("------------------------------------------")
    print(f"     V Angles")
    dt1 = pd.DataFrame(vangles_results, columns=columns)
    print(tabulate(dt1, headers='keys', tablefmt='psql', showindex=False))


def test_before_retraining(dataloader, model, loss_fn, no_samples=100, plot_predictions=False):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    compute_error_metrics = len(dataloader) == 1
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            if plot_predictions and batch == 0:
                plt.figure(figsize=(10, 6))
                plt.plot(np.arange(no_samples), y[:no_samples, 0])
                plt.plot(np.arange(no_samples), pred[:no_samples, 0])
                plt.xlabel('t')
                plt.ylabel('Voltage')
                plt.title(f'Voltage on node #1 ({no_samples} time instants)')
                plt.legend(['Actual', 'Estimated'])
                plt.show()
    test_loss /= num_batches
    if compute_error_metrics:
        mae = MeanAbsoluteError()
        mape = MeanAbsolutePercentageError()
        mse = MeanSquaredError()
        columns = ["MAE", "MAPE", "RMSE"]
        vmagnitudes_results = [[mae(pred[:, :caseNo], y[:, :caseNo]), mape(pred[:, :caseNo], y[:, :caseNo]),
                                torch.sqrt(mse(pred[:, :caseNo], y[:, :caseNo]))]]
        vangles_results = [[mae(pred[:, caseNo:], y[:, caseNo:]), None, torch.sqrt(mse(pred[:, caseNo:], y[:, caseNo:]))]]

        print("------------------------------------------")
        print(f"     V Magnitudes")
        dt1 = pd.DataFrame(vmagnitudes_results, columns=columns)
        print(tabulate(dt1, headers='keys', tablefmt='psql', showindex=False))

        print("------------------------------------------")
        print(f"     V Angles")
        dt1 = pd.DataFrame(vangles_results, columns=columns)
        print(tabulate(dt1, headers='keys', tablefmt='psql', showindex=False))
