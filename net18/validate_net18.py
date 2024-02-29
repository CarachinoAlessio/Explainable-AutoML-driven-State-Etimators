import random

import torch
import numpy as np
from torch.utils.data import DataLoader
import shap
from case118.networks import ANN
from net18.scenarios2 import get_data_by_scenario_and_case
from case118.dataset import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms


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

model = ANN(input_shape, 1524, num_classes, dropout=0.4656649267466235)
model.load_state_dict(torch.load("model_net18_53.pth"))
model.eval()
model = model.to(device)


VALID_DEBUG = list()

print('SCENARIO 1, CASE 1 VALIDATION')
s1_c1_data = get_data_by_scenario_and_case(1, 1)
x = s1_c1_data[0]
x_hat = s1_c1_data[1]
y = s1_c1_data[2]
y_hat = s1_c1_data[3]
sens = s1_c1_data[4]
'''
sens = sens[15]
'''


test_data = Dataset(x, y)
# test_dataloader = DataLoader(test_data, batch_size=100, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=len(test_data))

for batch, (X, y) in enumerate(test_dataloader):
    X, y = X.to(device), y.to(device)
    to_be_explained = X
    pred = model(X)
    y = y.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    VALID_DEBUG.append(y)
    VALID_DEBUG.append(pred)
    print(f'y: {y}\npred: {pred}')
    print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')

    '''
    alt_x = np.load('../nets/net_18_data/measured_data_x_alt.npy')
    alt_x = np.delete(alt_x, 18, axis=1)
    alt_x = np.delete(alt_x, 0, axis=1)
    alt_y = np.load('../nets/net_18_data/data_y_alt.npy')

    data = Dataset(alt_x, alt_y)
    dl = DataLoader(data, batch_size=500, drop_last=False, shuffle=True)

    (background, _) = next(iter(dl))
    background = background.to(device)

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(to_be_explained)
    relevance = abs(shap_values[15].ravel())
    norm_relevance = ((relevance - min(relevance)) / (max(relevance) - min(relevance)))

    # print(relevance)
    plt.imshow(norm_relevance.reshape((1, 53)))
    plt.colorbar()
    plt.show()
    plt.imshow(abs(sens.reshape((1, 53))))
    plt.colorbar()
    plt.show()
    '''




'''
# print('SCENARIO 1, CASE 2 VALIDATION')
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
        # print(f'y: {y}\npred: {pred}')
        print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')



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
        # print(f'y: {y}\npred: {pred}')
        print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')

'''

# print('SCENARIO 2, CASE 1 VALIDATION')
s1_c1_data = get_data_by_scenario_and_case(2, 1)
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
        # print(f'y: {y}\npred: {pred}')
        print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')
'''

print('SCENARIO 2, CASE 2 VALIDATION')
s1_c1_data = get_data_by_scenario_and_case(2, 2)
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
        # print(f'y: {y}\npred: {pred}')
        print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')



print('SCENARIO 2, CASE 3 VALIDATION')
s1_c1_data = get_data_by_scenario_and_case(2, 3)
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
        # print(f'y: {y}\npred: {pred}')
        print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')


'''

# print('SCENARIO 3, CASE 1 VALIDATION')
s1_c1_data = get_data_by_scenario_and_case(3, 1)
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
        # print(f'y: {y}\npred: {pred}')
        print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')

'''
print('SCENARIO 3, CASE 2 VALIDATION')
s1_c1_data = get_data_by_scenario_and_case(3, 2)
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
        # print(f'y: {y}\npred: {pred}')
        print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')



print('SCENARIO 3, CASE 3 VALIDATION')
s1_c1_data = get_data_by_scenario_and_case(3, 3)
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
        # print(f'y: {y}\npred: {pred}')
        print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')

'''


# print('SCENARIO 4, CASE 1 VALIDATION')
s1_c1_data = get_data_by_scenario_and_case(4, 1)
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
        # print(f'y: {y}\npred: {pred}')
        print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')
'''

print('SCENARIO 4, CASE 2 VALIDATION')
s1_c1_data = get_data_by_scenario_and_case(4, 2)
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
        # print(f'y: {y}\npred: {pred}')
        print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')



print('SCENARIO 4, CASE 3 VALIDATION')
s1_c1_data = get_data_by_scenario_and_case(4, 3)
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
        # print(f'y: {y}\npred: {pred}')
        print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')
'''


# print('SCENARIO 5, CASE 1 VALIDATION')
s1_c1_data = get_data_by_scenario_and_case(5, 1)
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
        # print(f'y: {y}\npred: {pred}')
        print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')

'''
print('SCENARIO 5, CASE 2 VALIDATION')
s1_c1_data = get_data_by_scenario_and_case(5, 2)
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
        # print(f'y: {y}\npred: {pred}')
        print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')
'''


print('SCENARIO 5, CASE 3 VALIDATION')
s1_c1_data = get_data_by_scenario_and_case(5, 3)
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
        VALID_DEBUG.append(y)
        VALID_DEBUG.append(pred)
        print(f'y: {y}\npred: {pred}')
        print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')


VALID_DEBUG = np.vstack((
    VALID_DEBUG[0],
    VALID_DEBUG[1],
    VALID_DEBUG[2],
    VALID_DEBUG[3],

))

print('debug')