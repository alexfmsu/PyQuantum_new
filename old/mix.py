# ---------------------------------------------------------------------------------------------------------------------
# system
from time import sleep
import sys
from math import sqrt
# ---------------------------------------------------------------------------------------------------------------------
# scientific
import numpy as np
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.TC_Lindblad_sink
from PyQuantum.Tools.Matrix import *
from PyQuantum.TCH.Cavity import Cavity
from PyQuantum.TCH.Unitary import Unitary
from Mix.Hamiltonian import Hamiltonian
import PyQuantum.Constants as Constants 
# from PyQuantum.TCL_sink.WaveFunction import WaveFunction
# from PyQuantum.TCL_sink.DensityMatrix import DensityMatrix

# from PyQuantum.TCL_sink.Evolution import run
import copy
# import Mix.config as config
from scipy.sparse import identity, kron, csc_matrix

from PyQuantum.TCH.Lindblad import operator_a, operator_L
# from PyQuantum.TCH.Lindblad import operator_a, operator_acrossa, operator_L
# from PyQuantum.TCH.Operators import operator_a, operator_acrossa, operator_L
# from PyQuantum.Common.Quantum.Operators import operator_a, operator_acrossa, operator_L
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.TC3
# from PyQuantum.TC_sink.states_collection import *
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.Tools
# from PyQuantum.Tools.LoadPackage import load_pkg
# from PyQuantum.Tools.Assert import *
# from PyQuantum.Tools.Print import hr
# from PyQuantum.Tools.MkDir import *
# from PyQuantum.Tools.CSV import *
# from PyQuantum.Tools.Units import *
from PyQuantum.Tools.Pickle import *
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.Common
# from PyQuantum.Common.Quantum.Operators import operator_a
# ---------------------------------------------------------------------------------------------------------------------
# exit(0)
# -----------------------------------------------
# l = config.g * 0.01
# cv = Cavity(wc=config.wc, wa=config.wa, g=config.g, n_atoms=config.n_atoms)

# cv.info()

# H = Hamiltonian(capacity=2, cavity=cv)

# H.print_states()

# H.print()
n_atoms = 2
wc = Constants.wc
wa = Constants.wc
# g = {
#     '0<->1': Constants.wc / 100
# }
# wa = wc

alpha = complex(1.0 / sqrt(3), 0)
beta = complex(sqrt(2) / sqrt(3), 0)
# beta = complex(1.0 / sqrt(2), 0)

ro_0 = [
    [        0,         0,                       0,                          0,          0],
    [        0,         0,                       0,                          0,          0],
    [        0,         0,           abs(alpha)**2,     alpha*beta.conjugate(),          0],
    [        0,         0,  beta*alpha.conjugate(),               abs(beta)**2,          0],
    [        0,         0,                       0,                          0,          0]
]

# for i in ro_0:
    # print(i)


_a = [
    [        0,         0,                       1,                          0,          0],
    [        0,         0,                       0,                          0,          0],
    [        0,         0,                       0,                          1,          0],
    [        0,         0,                       0,                          0,          0],
    [        0,         0,                       0,                          0,          0]
]

a = Matrix(m=len(_a), n=len(_a), dtype=np.complex128, data=_a)
# a.data = csc_matrix(_a)

across = Matrix(m=len(_a), n=len(_a), dtype=np.complex128, data = a.data.getH())
# across.data = a.data.getH()

# ro_t = copy.deepcopy(ro_0)

# ro_t = Matrix(m=len(ro_0), n=len(ro_0), dtype=np.complex128)
# ro_t.data = csc_matrix(ro_0)

# config.l = 0.01*config.g
# dt = 0.01 * 1e-9
l = Constants.wc
# l = 1.0 / dt

dt = 0.01 / l

# print(l / g)

l_out = {
    'L': a,
    'l': l
}

l_in = {
    'L': across,
    'l': l
}




# print(type(ro_t.data))
# U_conj.print()
# exit(0)


# print('-'*100)


# diag_abs = np.abs(ro_t.data.diagonal(), dtype=np.longdouble)

# max_energy = 1
# max_energy = diag_abs[2]+diag_abs[3]
# print(max_energy)
# exit(0)

g_list = []
time_list = []

g_0 = 0.01
g_step = 0.01
g_1 = 1.00+g_step

for g_ in np.round(np.arange(g_0, g_1, g_step), 2):
    # print(g_)
    # continue

    g_list.append(g_)

    cv = Cavity(wc=wc, wa=wa, g=Constants.wc * g_, n_atoms=n_atoms)
    # cv.info()

    H = Hamiltonian(capacity=2, cavity=cv)
    
    U = Unitary(H, dt)
    U_conj = U.conj()

    ro_t = Matrix(m=len(ro_0), n=len(ro_0), dtype=np.complex128, data=ro_0)
    
    # ---------------------------------------------------------------
    ro_t_data = ro_t.data.todense()

    for i in range(len(ro_t_data)):
        for j in range(len(ro_t_data)):
            print(np.round(abs(ro_t_data[i,j]), 3), '\t', end='')
        print()
    print()
    # ---------------------------------------------------------------

    L_out = operator_L(ro_t, l_out)
    L_in = operator_L(ro_t, l_in)

    L_op = L_out

    L_type = 'out'
    
    max_energy = 1
    sink = 0
    t = 0

    while True:
        diag_abs = np.abs(ro_t.data.diagonal(), dtype=np.longdouble)
        # print(abs(ro_t.data))

        if L_type == 'out':
            sink_prev = sink

            sink = (diag_abs[0])
            # sink = 1 - (diag_abs[2]+diag_abs[3])
            
            if sink < sink_prev:
                print(sink, '<', sink_prev)
                exit(0)

            if sink > 0.95:
                L_op = L_in
                L_type = 'in'
                max_energy = 0
                # exit(0)
        else:
            sink = diag_abs[2]+diag_abs[3]
            
            if sink > 0.95:
                max_energy = 1
                L_op = L_out
                L_type = 'out'
                sink = 0


        print(L_type, np.round(sink, 3))
        sleep(1)
        # for i in ro_t.data.toarray():
        #     print(i)
        #     print()
        #     sleep(2)
        #     for j in i:
        #         print(np.round(abs(j), 3), '\t', end='')
        #     print()
        # print('-'*100)
        d = ro_t.data.todense()

        all_zeros = True

        for i in range(len(d)):
            for j in range(len(d)):
                if i == j:
                    continue

                ampl = abs(d[i,j])

                if ampl >= 1e-3:
                    all_zeros = False
                    break

        # for i in range(len(d)):
        #     for j in range(len(d)):
        #         print(np.round(abs(d[i,j]), 4), '\t', end='')
        #     print()
        # print()

        if all_zeros:
            print("OK")
            print(t)
            time_list.append(t)
            break
            # exit(0)
        # print(type(ro_t.data))
        # sleep(0.1)

        ro_t.data = ((U.data).dot(ro_t.data + dt * L_op(ro_t).data)).dot(U_conj.data)
        # print(type(ro_t))
        # exit(0)
        # ro_t.normalize()
        Assert(abs(1 - ro_t.abs_trace()) <= 1e-4, "ro is not normed: " + str(ro_t.abs_trace()))

        t += dt


pickle_dump(g_list, 'g.pkl')
pickle_dump(time_list, 't.pkl')
    # print(t)


# print(l['L'].data)
    # state['w0'].normalize()

    # ro_0 = DensityMatrix(state['w0'])

# ro_0.print()
# T = 1 * config.ms

# # dt = 0.01 / l
# # dt = 1 * config.ns
# dt = 10 * config.ns
# # dt = 1 * config.ns / 10
# nt = int(T/dt)
# # dt = (0.001/l)

# # Assert(dt <= 0.01/l, 'dt > 0.01/l')

# nt = int(T/dt)

# cprint('T:', 'green', end='')
# print(time_unit_full(T))

# cprint('dt:', 'green', end='')
# # print(time_unit_full(dt))

# cprint('nt:', 'green', end='')
# print(nt)
# # -----------------------------------------------
# H_1_00 = get_H_1_00()
# w0_1_00 = get_w0_1_00(H_1_00)

# H_1_D = get_H_1_D()
# w0_1_D = get_w0_1_D(H_1_D)
# -----------------------------------------------
# w0_1_D.print()
# print()
# w0_1_00.print()
# exit(0)
# ---------------------------------------------------------------------------------------------------------------------
# mkdir('sink')
# mkdir('sink/1ms_l001g')
# ---------------------------------------------------------------------------------------------------------------------

# for state in [
#     {
#         'name': '1_00',
#         'w0': w0_1_00,
#         'H': H_1_00,
#     },
#     {
#         'name': '1_D',
#         'w0': w0_1_D,
#         'H': H_1_D,
#     },
# ]:
#     # -----------------------------------------------------------------------------------------------------------------
#     state['w0'].normalize()

#     ro_0 = DensityMatrix(state['w0'])

#     T_list = []
#     sink_list = []

#     run({
#         "ro_0": ro_0,
#         "H": state['H'],
#         "dt": dt,
#         "sink_list": sink_list,
#         "T_list": T_list,
#         "precision": 1e-3,
#         'sink_limit': 1,
#         "thres": 0.001,
#         'lindblad': {
#             'out': {
#                 'L': operator_a(state['H']),
#                 'l': l
#             },
#         },
#     })

#     # MkDir('sink')
#     pickle_dump(T_list, 'sink/1ms_l001g/T_list_' + state['name'] + '.pkl')
#     pickle_dump(sink_list, 'sink/1ms_l001g/sink_list_' +
#                 state['name'] + '.pkl')
#     # -----------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------

# =====================================================================================================================
