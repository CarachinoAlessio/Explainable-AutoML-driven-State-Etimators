import pandapower as pp


def Network_18_nodes_data():
    net = pp.create_empty_network(name="18bus_grid")
    pp.create_buses(net, 18, 11, min_vm_pu=0.95, max_vm_pu=1.05)
    pp.create_ext_grid(net, 0)

    idx = [1, 2, 4, 8, 9, 11, 16, 17]
    P = [0.1, 0.3, 0.4, 0.8, 0.8, 0.4, 0.1, 0.5]
    Q = [0.1, 0.1, 0.2, 0.4, 0.4, 0.2, 0.1, 0.3]
    pp.create_loads(net, idx, p_mw=P, q_mvar=Q)
    idx = [6, 13]
    P = [2.0, 5.0]
    Q = [0.6, 2.0]
    pp.create_sgens(net, idx, P, Q)
    fbus = [0, 1, 2, 3, 4, 5, 5, 7, 8, 9, 10, 10, 12, 3, 14, 15, 15]
    tbus = list(range(1, 18))
    length_km = 1
    R = [0.0, 0.0174, 0.0001, 0.0052, 0.0003, 0.0010, 0.0017, 0.0022, 0.0001, 0.0016, 0.0007, 0.0299, 0.0010, 0.0025,
         0.0011, 0.0034, 0.0013]
    X = [0.1, 0.0085, 0.0001, 0.0028, 0.0002, 0.0010, 0.0008, 0.0011, 0.0000, 0.0008, 0.0003, 0.0081, 0.0010, 0.0007,
         0.0003, 0.0009, 0.0004]
    R = [k * 12.1 for k in R]
    X = [k * 12.1 for k in X]
    C = 0
    Imax = 0.001
    pp.create_lines_from_parameters(net, fbus, tbus, length_km, R, X, C, Imax)
    return net
