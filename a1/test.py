import torch
import matplotlib.pyplot as plt
import numpy as np

device = "cpu"


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            if batch == 0:
                plt.plot(np.arange(y.shape[0]), y[:, 0])
                plt.plot(np.arange(y.shape[0]), pred[:, 0])
                plt.xlabel('t')
                plt.ylabel('Voltage')
                plt.title('Voltage on node #1')
                plt.legend(['Actual', 'Estimated'])
                plt.show()
    test_loss /= num_batches

