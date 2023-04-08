'''https://github.com/nbhusal/Power-System-State-Estimation'''

from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

SEED=1234
import numpy as np
import math
from matplotlib import pyplot as plt
np.random.seed(SEED)
import keras
from keras import backend as K
import tensorflow as tf
import os, shutil, scipy.io
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, add, Dropout, Lambda, Input, average, LSTM
from keras.layers.normalization import BatchNormalization
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras import optimizers
from keras import regularizers

from os import makedirs
from numpy import dstack, mean, std, argmax, tensordot, array
from numpy.linalg import norm
from keras.layers.merge import concatenate
from keras.utils import plot_model
from scipy.optimize import differential_evolution
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import timeit

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, add, Dropout, Lambda, LSTM, Flatten
from keras.layers import Input, average, TimeDistributed, SimpleRNN, LeakyReLU, GlobalAveragePooling1D
from keras.layers.convolutional import Conv1D, MaxPooling1D, AveragePooling1D
from keras import backend as K
from keras.layers.normalization import BatchNormalization


# # configure args


# data loading part
caseNo = 118
weight_4_mag = 100
weight_4_ang = 180/math.pi

psse_data = scipy.io.loadmat('data_for_SE_case'+str(caseNo)+'_for_ML.mat')
#print(psse_data['inputs'].shape, psse_data['labels'].shape)

data_x = psse_data['Z_measure']
data_y = psse_data['Output']
print(data_x.shape)
data_x=data_x[0:data_y.shape[0], :] # this is because measurement are available for 4 and half years but state estimation is available only for two years for IEEE 118
print(data_x.shape)
# print(data_x[0, -11:])
# data_x[:, -11:]=data_x[:, -11:]*weight_4_ang  # use this if angle measured need to convert into angle may need to improve the power
# scale the mags,
data_y[:, 0:caseNo] = weight_4_mag*data_y[:, 0:caseNo]
data_y[:, caseNo:] = weight_4_ang*data_y[:, caseNo:]


# separate them into training 40%, test 60%
split_train = int(0.6*data_x.shape[0])
test_x=data_x[:split_train, :]
test_y=data_y[:split_train, :]
train_x=data_x[split_train:, :]
train_y=data_y[split_train:, :]



# calculate the hubber loss
def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = K.abs(error) < clip_delta
    squared_loss = 0.5 * K.square(error)
    linear_loss  = clip_delta * (K.abs(error) - 0.5 * clip_delta)
    return tf.where(cond, squared_loss, linear_loss)
# calculate the hubber loss mean for the compiler
def huber_loss_mean(y_true, y_pred):
    return K.mean(huber_loss(y_true, y_pred))

# #Train the model
input_shape = (train_x.shape[1],)
num_classes=train_y.shape[1]


def nn1_psse(input_shape, num_classes, weights=None):
    '''
    :param input_shape:
    :param num_classes:
    :param weights: 6 layers
    :return: 1 hidden layer NN model with specified training loss
    '''
    data = Input(shape=input_shape, dtype='float', name='data')
    dense1 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(data)
    dense2 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense1)
    dense3 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense2)
    dense4 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense3)
    dense5 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense4)

    dense6 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense5)
#    dense7 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense6)
#    dense8 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense7)
#    dense9 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense8)
#    dense10 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense9)

#    drop1 = Dropout(rate=0.5, name='drop1')(dense8)
    predictions = Dense(units = num_classes, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense6)

    model = Model(inputs=data, outputs=predictions)
    if weights is not None:
        model.load_weights(weights)
    sgd = optimizers.adam(lr=0.001)

#    sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss=huber_loss_mean,
                  metrics=['mae'])

    return model


#
# epoch_num = 200
# # create directory for models
# makedirs('States_MLP_14') # For CNN

#
# def fit_model_DNN(train_x, train_y, input_shape, num_classes):
#     psse_model = nn1_psse(input_shape, num_classes, weights=None) # For DNN
#     psse_model.fit(train_x, train_y, epochs=200, batch_size=64, verbose=2)
#     return psse_model
#
#
# # fit and save model models to use later for ensembling
# n_members=6
# for i in range(n_members):
#     # fit model
#     model = fit_model_DNN(train_x, train_y, input_shape, num_classes)
#     # save model
#     filename = 'States_MLP_118/model_' + str(i + 1) + '.h5'  # for DNN
#     model.save(filename)
#     print('> Saved model %s' % filename)


# load models from file
def load_all_models(n_models):
    all_models=list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = 'States_MLP_'+str(caseNo)+'/model_' + str(i + 1) + '.h5'  # for Combined
        # load model from file
        #model=load_model(filename) # use this for MLP
        model=load_model(filename, custom_objects={'huber_loss_mean':huber_loss_mean})
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models
#
# #create stacked model input dataset as outputs from ensemble
# def stacked_dataset(members, inputX):
#     stackX=None
#     for model in members:
#         # make prediction
#         yhat=model.predict(inputX, verbose=0)
#         # stacke predictions into [row, members, probabilities]
#         if stackX is None:
#             stackX=yhat
#         else:
#             stackX=dstack((stackX, yhat))
#         # flatten prediction to [rows, members*probabilities]
#     stackX=stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
#     return stackX
#
#
# # fit a model based on the outputs from the ensemble members
# def fit_stacked_model(members, inputX, inputy):
#     # create dataset using ensembles
#     stackedX=stacked_dataset(members, inputX)
#     # fit standalone model
#     #model=MLPRegressor()
#     #model=KNeighborsRegressor()
#     #model=DecisionTreeRegressor()
#     model = LinearRegression()
#     model.fit(stackedX, inputy)
#     return model
#
# # make prediction with the stacked model
# def stacked_prediction(members, model, inputX):
#     # create dataset using ensemble
#     stackedX=stacked_dataset(members, inputX)
#     # make a prediction
#     yhat=model.predict(stackedX)
#     return yhat
#

# load all models
n_members=6
members=load_all_models(n_members)
print('Loaded %d models' %len(members))

# meta training dataset
# separate them into training meta learner 60%, test 40% of the remaining dataset (i.e. of train_x,
split_train = int(0.6*test_x.shape[0])
mtrain_x=test_x[:split_train, :]
mtrain_y=test_y[:split_train, :]
test_x=test_x[split_train:, :]
test_y=test_y[split_train:, :]

#data_new=scipy.io.loadmat('States_onli_SE_case'+str(caseNo)+'_for_ML.mat')
data_new=scipy.io.loadmat('States_onli_SE_case'+str(caseNo)+'_for_ML_Non_Gaussian.mat')
data_testy=data_new['Output']
data_testx=data_new['Z_measure']

data_testy[:, 0:caseNo] = weight_4_mag*data_testy[:, 0:caseNo]
data_testy[:, caseNo:] = weight_4_ang*data_testy[:, caseNo:]

test_x=data_testx[0:data_testy.shape[0], :]
test_y=data_testy

# evaluate standalone model on test dataset
vrmse=100
for model in members:
    start = timeit.default_timer()
    yhat = model.predict(test_x)
    stop = timeit.default_timer()
    # print('Time: %f' % (stop - start))
    print('Per iteration time: %f' % ((stop - start) / test_x.shape[0] * 1000))


    RMSEv = np.sqrt(mean_squared_error(yhat[:, 0:caseNo], test_y[:, 0:caseNo]))
    print('Model Test data RMSEv: %.4f' % RMSEv)
    MAEv = mean_absolute_error(yhat[:, 0:caseNo], test_y[:, 0:caseNo])
    print('Model Test data MAEv: %.4f' % MAEv)

    RMSEa = np.sqrt(mean_squared_error(yhat[:, caseNo:], test_y[:, caseNo:]))
    print('Model Test data RMSEa: %.4f' % RMSEa)
    MAEa = mean_absolute_error(yhat[:, caseNo:], test_y[:, caseNo:])
    print('Model Test data MAEa: %.4f' % MAEa)

    if vrmse > RMSEv:
        yhat_save = yhat
        vrmse = RMSEv
        vmae = MAEv
        armse = RMSEa
        amae = MAEa

print('RMSE to win min model for voltage is : %.4f' % vrmse)
print('MAE to win min model for voltage is : %.4f' % vmae)
print('RMSE to win min model for phase is : %.4f' % armse)
print('MAE to win min model for phase is : %.4f' % amae)