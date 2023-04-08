import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.plotting import simple_plot, simple_plotly, plot_loading
from pandapower.networks import create_cigre_network_lv

net = pp.create_empty_network()

b1 = pp.create_bus(net, name="bus 1", vn_kv=1., index=1)
b2 = pp.create_bus(net, name="bus 2", vn_kv=1., index=2)
b3 = pp.create_bus(net, name="bus 3", vn_kv=1., index=3)

pp.create_ext_grid(net, 1)  # set the slack bus to bus 1

l1 = pp.create_line_from_parameters(net, 1, 2, 1, r_ohm_per_km=.01, x_ohm_per_km=.03, c_nf_per_km=0., max_i_ka=1)
l2 = pp.create_line_from_parameters(net, 1, 3, 1, r_ohm_per_km=.02, x_ohm_per_km=.05, c_nf_per_km=0., max_i_ka=1)
l3 = pp.create_line_from_parameters(net, 2, 3, 1, r_ohm_per_km=.03, x_ohm_per_km=.08, c_nf_per_km=0., max_i_ka=1)

print(net)
cigre = create_cigre_network_lv()
print(cigre)
simple_plot(cigre)