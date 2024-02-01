import torch
import torch.nn as nn
import time

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import shap
import numpy as np
import matplotlib.pyplot as plt
import parser
from case118.dataset import Dataset
from case118.networks import ANN
from case118.train import train
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
import pandas as pd
from tabulate import tabulate
import seaborn as sns

from net18.scenarios2 import get_data_by_scenario_and_case

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
device = "cpu"


def test(dataloader, model, loss_fn, plot_predictions=False):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # pred /= (1-0.2)
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
    return torch.sqrt(mse(pred, y))

'''
data_x = np.load('../nets/net_18_data/measured_data_x.npy')
data_x = np.delete(data_x, 18, axis=1)
data_x = np.delete(data_x, 0, axis=1)

data_y = np.load('../nets/net_18_data/data_y.npy')
#measured_x = np.load('../nets/net_18_data/measured_data_x.npy')
#measured_y = np.load('../nets/net_18_data/measured_data_y.npy')
'''
stable_x = np.load('../nets/net_18_data/measured_data_x_stable.npy')
stable_y = np.load('../nets/net_18_data/data_y_stable.npy')

alt_x = np.load('../nets/net_18_data/measured_data_x_alt.npy')
alt_y = np.load('../nets/net_18_data/data_y_alt.npy')

#data_x = np.vstack((stable_x, alt_x))
data_x = alt_x
data_y = alt_y
#data_y = np.vstack((stable_y, alt_y))

s1_c1_data = get_data_by_scenario_and_case(1, 1)
x = s1_c1_data[0].ravel()

'''
for j, i in enumerate(data_x.T):
    # if j < 30:
        continue
    sns.displot(i, kde=True, bins=50).set(title=str(j))
    plt.axvline(x=x[j], color='r', linestyle='-')
    plt.tight_layout(pad=3.0)
    #plt.savefig(str(j))
    plt.show()
'''


#measured_x = np.load('../nets/net_18_data/measured_data_x.npy')
#measured_y = np.load('../nets/net_18_data/measured_data_y.npy')

if verbose:
    print(data_x.shape)
    print(data_y.shape)

# separate them into training 60%, test 40%
split_train = int(0.8 * data_x.shape[0])
train_x = data_x[:split_train, :]
train_y = data_y[:split_train, :]

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, shuffle=True)

train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.3, shuffle=True)

# test_x = data_x[split_train:, :]
# test_y = data_y[split_train:, :]


if verbose:
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    print(val_x.shape)
    print(val_y.shape)

train_data = Dataset(train_x, train_y)
train_dataloader = DataLoader(train_data, batch_size=16, drop_last=True)

val_data = Dataset(val_x, val_y)
val_dataloader = DataLoader(val_data, batch_size=int(len(val_data) / s), drop_last=False)

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
#optimizer = torch.optim.RMSprop(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)


losses = []
torch.backends.cudnn.benchmark = True
if train_time or True:
    epochs = 30
    root_mse = 100
    training_exception = True
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        l = train(train_dataloader, model, loss_fn, optimizer)
        losses.append(l)
        if t > epochs - 30:
            if (new_root_mse := test(test_dataloader, model, loss_fn, plot_predictions=False)) < root_mse:
                training_exception = False
                print("New improved model found! Saving...")
                root_mse = new_root_mse
                torch.save(model.state_dict(), "model_net18_53.pth")
                torch.save(optimizer.state_dict(), "optimizer_53.pth")
                print("Saved.")
    if training_exception:
        raise ("Found only bad models in the last 10 epochs. Please check training parameters.")
    print("Done!")

plt.plot(losses)
plt.show()
print("Testing the best model...")
model.load_state_dict(torch.load("model_net18_53.pth"))
model.eval()
test(test_dataloader, model, loss_fn, plot_predictions=True)



(X, Y) = next(iter(test_dataloader))
X = X.to(device)

measurements = X
estimations = model(measurements).detach().numpy()
background = measurements[0:100].to(device)
# background = shap.maskers.Independent(measurements.numpy(), max_samples=100)
to_be_explained = measurements[100:101].to(device)

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(to_be_explained)
print("Saving shap_values on file...")
with open('shap_values_net18_SHAP.npy', 'wb') as f:
    np.save(f, np.array(shap_values))

print("Loading shap_values from file...")
with open('shap_values_net18_SHAP.npy', 'rb') as f:
    shap_values = list(np.load(f))

relevance = abs(shap_values[0].ravel())


norm_relevance = ((relevance - min(relevance)) / (max(relevance) - min(relevance)))
'''
print(relevance)
plt.imshow(norm_relevance.reshape((5, 11)))
plt.colorbar()
'''

'''
fig, ax = plt.subplots()
labels = ['p_mw', 'q_mvar', 'vm_pu', 'p_mw_lines', 'q__mvar_lines']
aggregate_data = [sum(relevance[:18])/18., sum(relevance[18:36])/18., sum(relevance[36:41])/5., sum(relevance[41:48])/7., sum(relevance[48:])/7.]
ax.pie(aggregate_data, labels=labels, autopct='%1.1f%%')
plt.show()
'''


p_indices = range(18)
q_indices = range(18)
v_bus_indices = [0, 3, 5, 10, 15]
p_indices_lines = [0, 3, 6, 10, 11, 13, 15]
q_indices_lines = [0, 3, 6, 10, 11, 13, 15]
positioning = dict()
positioning.setdefault('p_mw', 0)
positioning.setdefault('q_mvar', 18)
positioning.setdefault('vm_pu', 36)
positioning.setdefault('p_mw_lines', 41)
positioning.setdefault('q__mvar_lines', 48)


labels = [str(i) for i in range(18)]
aggregate_data = []
for i in range(18):
    data = 0.0
    for k in positioning.keys():
        match k:
            case 'p_mw':
                data += relevance[positioning[k]+i]
            case 'q_mvar':
                data += relevance[positioning[k]+i]
            case 'vm_pu':
                if i in v_bus_indices:
                    data += relevance[positioning[k] + v_bus_indices.index(i)]
            case 'p_mw_lines':
                if i in p_indices_lines:
                    data += relevance[positioning[k] + p_indices_lines.index(i)]
            case _:
                if i in q_indices_lines:
                    data += relevance[positioning[k] + q_indices_lines.index(i)]
    aggregate_data.append(data)


sorted_index = np.asarray(aggregate_data).argsort()[:12:-1]
labels = list(np.asarray(labels)[sorted_index]) + ['altro']
aggregate_data = list(np.asarray(aggregate_data)[sorted_index]) + [sum([j for i,j in enumerate(aggregate_data) if i not in sorted_index])]

fig, ax = plt.subplots()
ax.pie(aggregate_data, labels=labels, autopct='%1.1f%%')




plt.show()

print("--------- Done ---------")



