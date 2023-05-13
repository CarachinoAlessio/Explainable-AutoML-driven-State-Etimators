from my_networks import Network_18_nodes_data
from pandapower.plotting import simple_plot
import pandapower as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

net = Network_18_nodes_data()
# simple_plot(net, plot_sgens=True, plot_loads=True)
print(net)

# definisci i profili di carico
# load_profiles_p = pd.DataFrame(np.random.rand(8, len(net.bus)), index=range(8), columns=net.bus.index)
# load_profiles_q = pd.DataFrame(np.random.rand(8, len(net.bus)), index=range(8), columns=net.bus.index)

load_profiles_p = 0.5 + np.random.rand(256, len(net.load))
load_profiles_q = 0.5 + np.random.rand(256, len(net.load))

# applica i profili di carico alla rete
pp.runpp(net)

# print(net.res_bus)
res_p_mw = list([net.res_bus["p_mw"].values])
res_q_mvar = list([net.res_bus["q_mvar"].values])

measured_p_mw = list(np.random.normal([net.res_bus["p_mw"].values], 0.003))
measured_q_mvar = list(np.random.normal([net.res_bus["q_mvar"].values], 0.003))

v_bus_indices = [0, 3, 5, 10, 15]
res_vm_pu = list([net.res_bus["vm_pu"].values[v_bus_indices]])
measured_vm_pu = list(np.random.normal([net.res_bus["vm_pu"].values[v_bus_indices]], 0.003))

p_q_indices = [0, 3, 6, 10, 11, 13, 15]
res_p_mw_lines = list([net.res_line["p_from_mw"].values[p_q_indices]])
res_q_mvar_lines = list([net.res_line["q_from_mvar"].values[p_q_indices]])
measured_p_mw_lines = list(np.random.normal([net.res_line["p_from_mw"].values[p_q_indices]], 0.003))
measured_q_mvar_lines = list(np.random.normal([net.res_line["q_from_mvar"].values[p_q_indices]], 0.003))

# res = list([net.res_bus["vm_pu"].values])

for p_factor, q_factor in zip(load_profiles_p, load_profiles_q):

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

    measured_p_mw.append(np.random.normal([net.res_bus["p_mw"].values], 0.003).flatten())
    measured_q_mvar.append(np.random.normal([net.res_bus["q_mvar"].values], 0.003).flatten())
    measured_vm_pu.append(np.random.normal([net.res_bus["vm_pu"].values[v_bus_indices]], 0.003).flatten())
    measured_p_mw_lines.append(np.random.normal([net.res_line["p_from_mw"].values[p_q_indices]], 0.003).flatten())
    measured_q_mvar_lines.append(np.random.normal([net.res_line["q_from_mvar"].values[p_q_indices]], 0.003).flatten())

    # print(net.res_bus)
    # res.append(net.res_bus["vm_pu"].values)

'''
plt.plot(np.array(measured_q_mvar_lines).T[0])
plt.plot(np.array(res_q_mvar_lines).T[0], linestyle='--', marker='.')
plt.legend(['Meas', 'Act'])
plt.show()
'''

data_x = np.hstack((res_p_mw, res_q_mvar, res_p_mw_lines, res_q_mvar_lines))
measured_data_x = np.hstack((measured_p_mw, measured_q_mvar, measured_p_mw_lines, measured_q_mvar_lines))
data_y = res_vm_pu
measured_data_y = measured_vm_pu

np.save('./net_18_data/data_x.npy', data_x)
np.save('./net_18_data/data_y.npy', data_x)
np.save('./net_18_data/measured_data_x.npy', measured_data_x)
np.save('./net_18_data/measured_data_y.npy', measured_data_y)
