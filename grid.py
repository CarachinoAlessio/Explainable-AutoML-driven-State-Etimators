# Youtube Tutorial Machine Learning 1: https://www.youtube.com/watch?v=kHbAMoCv-d4
import simbench as sb
import matplotlib.pyplot as plt
import pandapower.timeseries as ts
from pandapower.control.controller.const_control import ConstControl
from pandapower.timeseries.data_sources.frame_data import DFData
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from time import time

'''
grid_code = "1-HV-urban--0-sw"
net = sb.get_simbench_net(grid_code)
profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)

print(profiles.keys())
print(net)

sgen_p = profiles[("sgen", "p_mw")]
load_p = profiles[("load", "p_mw")]
load_q = profiles[("load", "q_mvar")]
print(sgen_p)


load_p.sum(axis=1).plot(label="load")
sgen_p.sum(axis=1).plot(label="sgen")
plt.legend()
plt.show()

ds = DFData(sgen_p)
ConstControl(net, "sgen", "p_mw", element_index=net.sgen.index, profile_name=sgen_p.columns, data_source=ds)
ds = DFData(load_p)
ConstControl(net, "load", "p_mw", element_index=net.load.index, profile_name=load_p.columns, data_source=ds)
ds = DFData(load_q)
ConstControl(net, "load", "q_mvar", element_index=net.load.index, profile_name=load_q.columns, data_source=ds)

ts.OutputWriter(net, output_path="./", output_file_type=".json")
ts.run_timeseries(net)
'''

'''
grid_code = "1-HV-urban--0-sw"
net = sb.get_simbench_net(grid_code)
profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)

sgen_p = profiles[("sgen", "p_mw")]
load_p = profiles[("load", "p_mw")]
load_q = profiles[("load", "q_mvar")]

X = pd.concat([sgen_p, load_p, load_q], axis=1)
y = pd.read_json("./res_line/loading_percent.json")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print(X.shape)
print(X_train.shape)
print(X_test.shape)
'''

# sgen_p, load_p, load_q
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
# line loading results
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)

ann = MLPRegressor(verbose=1)
# 10% of power flow data
ann.fit(X_train, y_train)
# 90% of power flow data
y_predict = ann.predict(X_test)
y_predict = scaler.inverse_transform(y_predict)

plt.plot(y_test[:96, 53], alpha=.5, linestyle="--", label="correct line loading values")
plt.plot(y_predict[:96, 53], alpha=.5, linestyle="-", label="predicted line loading values")
plt.legend()
plt.show()


mse = mean_squared_error(y_test,  y_predict)
print(f"the error is only {mse:.2f}%")


t0 = time()
y_predict = ann.predict(X_test)
t1 = time() - t0
print(f"ANN time: {t1:.2f}")