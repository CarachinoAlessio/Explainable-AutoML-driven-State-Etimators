import torch
import torch.nn as nn
import scipy
import math

import tensorflow as tf
from keras import metrics
from tensorflow import keras


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


data_x = np.load('../nets/net_18_data/measured_data_x.npy')
data_y = np.load('../nets/net_18_data/data_y.npy')
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

train_data = Dataset(train_x, train_y)
train_dataloader = DataLoader(train_data, batch_size=16, drop_last=False)

val_data = Dataset(val_x, val_y)
val_dataloader = DataLoader(val_data, batch_size=int(len(val_data) / s), drop_last=False)

test_data = Dataset(test_x, test_y)
# test_dataloader = DataLoader(test_data, batch_size=100, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=len(test_data))

# Train the model
input_shape = train_x.shape[1]
num_classes = train_y.shape[1]

#model = ANN(input_shape, 2500, num_classes).to(device)
model = keras.Sequential([
    keras.layers.Input(shape=input_shape),
    keras.layers.Dense(2500, activation='relu'),
    keras.layers.Dense(num_classes)
])

if verbose:
    print(model)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='mean_squared_error',
              metrics=[metrics.MeanAbsoluteError()])

num_epochs = 30
batch_size = 32

model.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size)
test_loss, test_accuracy = model.evaluate(test_x, test_y)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

batch = next(iter(test_dataloader))
measurements, _ = batch
measurements = measurements.detach().numpy()
estimations = model.predict(measurements)
background = measurements[:100]
shap_values_time = True
if shap_values_time:
    to_be_explained = measurements[100:150]

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(to_be_explained)

    shap_values_node_1 = shap_values[0]
    #shap.summary_plot(shap_values_node_1, plot_type="bar", max_display=12)
    #shap.summary_plot(shap_values_node_1, plot_type="violin", max_display=12)
    #t = dict()
    #t.setdefault("values", shap_values_node_1)
    shap.plots.heatmap(shap_values)



print("--------- Done ---------")



