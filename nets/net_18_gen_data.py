from my_networks import Network_18_nodes_data
from pandapower.plotting import simple_plot
import pandapower as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

net = Network_18_nodes_data()
# simple_plot(net, plot_sgens=True, plot_loads=True)
print(net)


# 600 timestep
load_profiles_p = 0.5 + np.random.rand(600, len(net.load))
load_profiles_q = 0.5 + np.random.rand(600, len(net.load))

pp.runpp(net)


res_p_mw = list([net.res_bus["p_mw"].values])
res_q_mvar = list([net.res_bus["q_mvar"].values])

v_bus_indices = [0, 3, 5, 10, 15]
res_vm_pu = list([net.res_bus["vm_pu"].values[v_bus_indices]])

res_all_vm_pu = list([net.res_bus["vm_pu"].values])

p_q_indices = [0, 3, 6, 10, 11, 13, 15]
res_p_mw_lines = list([net.res_line["p_from_mw"].values[p_q_indices]])
res_q_mvar_lines = list([net.res_line["q_from_mvar"].values[p_q_indices]])


for p_factor, q_factor in tqdm(zip(load_profiles_p, load_profiles_q), total=600):

    for i in range(len(net.load.p_mw)):
        net.load.p_mw[i] = net.load.p_mw[i] * p_factor[i]
        net.load.q_mvar[i] = net.load.q_mvar[i] * q_factor[i]

    pp.runpp(net)

    # for i in range(len(net.res_bus["vm_pu"].values)):
    #    pp.create_measurement(net, 'p', 'bus', net.res_bus["vm_pu"].values[i], 0.003, i)

    res_p_mw.append(net.res_bus["p_mw"].values)
    res_q_mvar.append(net.res_bus["q_mvar"].values)
    res_vm_pu.append(net.res_bus["vm_pu"].values[v_bus_indices])
    res_p_mw_lines.append(net.res_line['p_from_mw'].values[p_q_indices])
    res_q_mvar_lines.append(net.res_line['q_from_mvar'].values[p_q_indices])
    res_all_vm_pu.append(net.res_bus["vm_pu"].values)


'''
plt.plot(np.array(measured_q_mvar_lines).T[0])
plt.plot(np.array(res_q_mvar_lines).T[0], linestyle='--', marker='.')
plt.legend(['Meas', 'Act'])
plt.show()
'''

# agguingo rumore introdotto dallo strumento di misura
measured_p_mw = list(np.random.normal(res_p_mw, 0.003))
measured_q_mvar = list(np.random.normal(res_q_mvar, 0.003))
measured_vm_pu = list(np.random.normal(res_vm_pu, 0.003))
measured_all_vm_pu = list(np.random.normal(res_all_vm_pu, 0.003))
measured_p_mw_lines = list(np.random.normal(res_p_mw_lines, 0.003))
measured_q_mvar_lines = list(np.random.normal(res_q_mvar_lines, 0.003))


data_x = np.hstack((res_p_mw, res_q_mvar, res_vm_pu, res_p_mw_lines, res_q_mvar_lines))
measured_data_x = np.hstack((measured_p_mw, measured_q_mvar, measured_vm_pu, measured_p_mw_lines, measured_q_mvar_lines))
data_y = res_all_vm_pu
measured_data_y = measured_all_vm_pu

np.save('./net_18_data/data_x.npy', data_x)
np.save('./net_18_data/data_y.npy', data_y)
np.save('./net_18_data/measured_data_x.npy', measured_data_x)
np.save('./net_18_data/measured_data_y.npy', measured_data_y)
