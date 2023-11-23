import torch
import numpy as np
from torch.utils.data import DataLoader

from case118.networks import ANN
from net18.scenarios import get_data_by_scenario
from case118.dataset import Dataset



device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


data_x = np.load('../nets/net_18_data/measured_data_x.npy')
data_y = np.load('../nets/net_18_data/data_y.npy')
#measured_x = np.load('../nets/net_18_data/measured_data_x.npy')
#measured_y = np.load('../nets/net_18_data/measured_data_y.npy')
data_x = np.delete(data_x, 18, axis=1)
data_x = np.delete(data_x, 0, axis=1)

train_x = data_x
train_y = data_y


# Train the model
input_shape = data_x.shape[1]
num_classes = data_y.shape[1]

model = ANN(input_shape, 2500, num_classes).to(device)
model.load_state_dict(torch.load("model_net18_53.pth"))
model.eval()


scenario1 = get_data_by_scenario(1)
x = scenario1[0]
x_hat = scenario1[1]
y = scenario1[2]
y_hat = scenario1[3]

test_data = Dataset(x, y)
# test_dataloader = DataLoader(test_data, batch_size=100, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=len(test_data))

with torch.no_grad():
    for batch, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        print(pred)
