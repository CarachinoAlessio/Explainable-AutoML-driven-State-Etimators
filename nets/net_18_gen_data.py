from my_networks import Network_18_nodes_data
from pandapower.plotting import simple_plot
import pandapower as pp
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

net = Network_18_nodes_data()
# simple_plot(net, plot_sgens=True, plot_loads=True)
print(net)


def generate_numbers(time_instants, n_bus):
    res = np.zeros((n_bus, time_instants))
    for i in range(n_bus):
        # Definizione dei parametri
        intervallo = np.arange(600)
        frequenza = 0.05  # Regola la frequenza dell'andamento
        ampiezza = 1  # Regola l'ampiezza dell'andamento
        traslazione = 0.1  # Regola la traslazione verticale dell'andamento
        rumore = 0.2  # Regola la variazione casuale dell'andamento

        # Generazione dell'andamento
        andamento = ampiezza * np.sin(frequenza * intervallo) + traslazione
        andamento += rumore * np.random.normal(-1, 1, size=andamento.shape)
        andamento = abs(andamento) + .5
        res[i] = andamento

        '''
        plt.plot(intervallo, andamento)
        plt.xlabel('Tempo')
        plt.ylabel('Valore')
        plt.show()
        '''
    return res.T


# 600 timestep, ad ogni timestep i profili saranno aggiornati di un fattore moltiplicativo tra 0.5 e 1.5
original_p_values = copy.deepcopy(net.load.p_mw)
original_q_values = copy.deepcopy(net.load.q_mvar)
# load_profiles_p = 0.5 + np.random.rand(600, len(net.load))
# load_profiles_q = 0.5 + np.random.rand(600, len(net.load))
load_profiles_p = generate_numbers(600, len(net.load))
load_profiles_q = generate_numbers(600, len(net.load))

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
    # aggiorno load di p e q
    for i in range(len(net.load.p_mw)):
        net.load.p_mw[i] = original_p_values[i] * p_factor[i]
        net.load.q_mvar[i] = original_q_values[i] * q_factor[i]

    pp.runpp(net)

    # for i in range(len(net.res_bus["vm_pu"].values)):
    #    pp.create_measurement(net, 'p', 'bus', net.res_bus["vm_pu"].values[i], 0.003, i)

    res_p_mw.append(net.res_bus["p_mw"].values)
    res_q_mvar.append(net.res_bus["q_mvar"].values)
    res_vm_pu.append(net.res_bus["vm_pu"].values[v_bus_indices])
    res_p_mw_lines.append(net.res_line['p_from_mw'].values[p_q_indices])
    res_q_mvar_lines.append(net.res_line['q_from_mvar'].values[p_q_indices])
    res_all_vm_pu.append(net.res_bus["vm_pu"].values)

# agguingo rumore introdotto dallo strumento di misura
measured_p_mw = list(np.random.normal(res_p_mw, 0.003))
measured_q_mvar = list(np.random.normal(res_q_mvar, 0.003))
measured_vm_pu = list(np.random.normal(res_vm_pu, 0.003))
measured_all_vm_pu = list(np.random.normal(res_all_vm_pu, 0.003))
measured_p_mw_lines = list(np.random.normal(res_p_mw_lines, 0.003))
measured_q_mvar_lines = list(np.random.normal(res_q_mvar_lines, 0.003))

data_x = np.hstack((res_p_mw, res_q_mvar, res_vm_pu, res_p_mw_lines, res_q_mvar_lines))
measured_data_x = np.hstack(
    (measured_p_mw, measured_q_mvar, measured_vm_pu, measured_p_mw_lines, measured_q_mvar_lines))
data_y = res_all_vm_pu
measured_data_y = measured_all_vm_pu

np.save('./net_18_data/data_x.npy', data_x)
np.save('./net_18_data/data_y.npy', data_y)
np.save('./net_18_data/measured_data_x.npy', measured_data_x)
np.save('./net_18_data/measured_data_y.npy', measured_data_y)
