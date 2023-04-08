import torch
import torch.nn as nn
import scipy
import math
from torch.utils.data import DataLoader

from a1.dataset import Dataset
from a1.network import ANN
from a1.train import train
from a1.test import test

torch.manual_seed(42)
train_time = False

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
weight_4_ang = 180/math.pi

data = scipy.io.loadmat('data_for_SE_case'+str(caseNo)+'_for_ML.mat')
#print(data['inputs'].shape, data['labels'].shape)

data_x = data['Z_measure']
data_y = data['Output']

data_x = data_x[0:data_y.shape[0], :] # this is because measurement are available for 4 and half years but state estimation is available only for two years for IEEE 118

data_y[:, 0:caseNo] = weight_4_mag*data_y[:, 0:caseNo]
data_y[:, caseNo:] = weight_4_ang*data_y[:, caseNo:]
print(data_x.shape)
print(data_y.shape)

# separate them into training 40%, test 60%
split_train = int(0.6*data_x.shape[0])
test_x = data_x[:split_train, :]
test_y = data_y[:split_train, :]
train_x = data_x[split_train:, :]
train_y = data_y[split_train:, :]

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

train_data = Dataset(train_x, train_y)
train_dataloader = DataLoader(train_data, batch_size=100, drop_last=True)

test_data = Dataset(test_x, test_y)
test_dataloader = DataLoader(test_data, batch_size=100, drop_last=True)


# Train the model
input_shape = train_x.shape[1]
num_classes = train_y.shape[1]

model = ANN(input_shape, 2500, num_classes).to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

if train_time:
    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        #test(test_dataloader, model, loss_fn)
    print("Done!")
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

else:
    model.load_state_dict(torch.load("model.pth"))
    test(test_dataloader, model, loss_fn)








