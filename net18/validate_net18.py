import torch
import numpy as np
from torch.utils.data import DataLoader

from case118.networks import ANN
from net18.scenarios import get_data_by_scenario_and_case
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



print('SCENARIO 1, CASE 1 VALIDATION')
s1_c1_data = get_data_by_scenario_and_case(1, 1)
x = s1_c1_data[0]
x_hat = s1_c1_data[1]
y = s1_c1_data[2]
y_hat = s1_c1_data[3]

test_data = Dataset(x, y)
# test_dataloader = DataLoader(test_data, batch_size=100, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=len(test_data))

with torch.no_grad():
    for batch, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        y = y.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        print(f'y: {y}\npred: {pred}\nstd: {np.sqrt(np.mean(np.square(y - pred)))}')



print('SCENARIO 1, CASE 2 VALIDATION')
s1_c1_data = get_data_by_scenario_and_case(1, 2)
x = s1_c1_data[0]
x_hat = s1_c1_data[1]
y = s1_c1_data[2]
y_hat = s1_c1_data[3]

test_data = Dataset(x, y)
# test_dataloader = DataLoader(test_data, batch_size=100, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=len(test_data))

with torch.no_grad():
    for batch, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        y = y.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        print(f'y: {y}\npred: {pred}\nstd: {np.sqrt(np.mean(np.square(y - pred)))}')



print('SCENARIO 1, CASE 3 VALIDATION')
s1_c1_data = get_data_by_scenario_and_case(1, 3)
x = s1_c1_data[0]
x_hat = s1_c1_data[1]
y = s1_c1_data[2]
y_hat = s1_c1_data[3]

test_data = Dataset(x, y)
# test_dataloader = DataLoader(test_data, batch_size=100, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=len(test_data))

with torch.no_grad():
    for batch, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        y = y.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        print(f'y: {y}\npred: {pred}\nstd: {np.sqrt(np.mean(np.square(y - pred)))}')

