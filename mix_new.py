# =====================================================================================================================
# system
import sys
import copy
from math import sqrt
from time import sleep
# ---------------------------------------------------------------------------------------------------------------------
# scientific
import numpy as np
from scipy.sparse import identity, kron, csc_matrix
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.TCH
from PyQuantum.TCH.Cavity import Cavity
from PyQuantum.TCH.Unitary import Unitary
from PyQuantum.TCH.Lindblad import operator_a, operator_L
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.Tools
from PyQuantum.Tools.Matrix import *
from PyQuantum.Tools.Pickle import *
from PyQuantum.Tools.Mkdir import *
from PyQuantum.Tools.to_Hz import *
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.Constants
import PyQuantum.Constants as Constants 
# ---------------------------------------------------------------------------------------------------------------------
from Mix.Hamiltonian import Hamiltonian
# =====================================================================================================================



# =====================================================================================================================
capacity = 2

n_atoms = 2

wc = Constants.wc
wa = Constants.wc

g_step = 0.001
# g_0 = 1
g_0 = 0.05
# g_0 = g_step
g_1 = 1
g_0 = g_1

l = Constants.wc / 100
# l = Constants.wc / 10000

dt = 0.001 / l
# dt = 0.01 / l

sink_limit = 0.95

sink_precision = 1e-4
ro_trace_precision = 1e-4
ampl_precision = 1e-3

path = 'l_wc10000'
mkdir(path)

info = {
    # ---------------------------------------
    'capacity': capacity,
    # ---------------------------------------
    'n_atoms': n_atoms,
    # ---------------------------------------
    'wc': wc,
    'wc(Hz)': to_Hz(wc),
    'wa': wa,
    'wa(Hz)': to_Hz(wa),
    # ---------------------------------------
    'g_0': g_0,
    'g_step': g_step,
    'g_1': g_1,
    # ---------------------------------------
    'l': l,
    'l(Hz)': to_Hz(l),
    'l/wc': l/wc,
    # ---------------------------------------
    'dt': dt,
    'dt(s)': time_unit_full(dt),
    'dt*l': dt*l,
    # ---------------------------------------
    'sink_limit': sink_limit,
    # ---------------------------------------
    'sink_precision': sink_precision,
    'ro_trace_precision': ro_trace_precision,
    'ampl_precision': ampl_precision,
    # ---------------------------------------
    'path': path,
    # ---------------------------------------
}

pickle_dump(info, path+'/info.pkl')
# =====================================================================================================================
# alpha = complex(1.0 / sqrt(3), 0)
# beta = complex(sqrt(2) / sqrt(3), 0)

alpha = complex(1.0 / sqrt(2), 0)
beta = complex(1.0 / sqrt(2), 0)

ro_0_ = [
    [        0,         0,                       0,                          0,          0],
    [        0,         0,                       0,                          0,          0],
    [        0,         0,           abs(alpha)**2,     alpha*beta.conjugate(),          0],
    [        0,         0,  beta*alpha.conjugate(),               abs(beta)**2,          0],
    [        0,         0,                       0,                          0,          0]
]


    #      |00〉      |0s〉                    |10〉                       |1s〉       |0t〉
_a = [
    [        0,         0,                       1,                          0,          0],
    [        0,         0,                       0,                          1,          0],
    [        0,         0,                       0,                          0,          0],
    [        0,         0,                       0,                          0,          0],
    [        0,         0,                       0,                          0,          0]
]
ro_0 = Matrix(m=len(ro_0_), n=len(ro_0_), dtype=np.complex128, data=lil_matrix(ro_0_))
a = Matrix(m=len(_a), n=len(_a), dtype=np.complex128, data=lil_matrix(_a))
across = a.conj()

# ro_0.print()
# exit(0)

l_out = {
    'L': a,
    'l': l
}

l_in = {
    'L': across,
    'l': l*1000
}
# print(1)


g_list = []
time_list = []
ampl01 = []

def add_ph(ro_t):
    for i in range(0, 2):
        for j in range(0, 2):
            ro_t.data[i+2, j+2], ro_t.data[i, j] = ro_t.data[i, j], ro_t.data[i+2, j+2]

    return ro_t
for g_ in np.round(np.arange(g_0, g_1+g_step, g_step), 3):
    print('g:', g_)
    g_list.append(g_)

    cv = Cavity(wc=wc, wa=wa, g=Constants.wc * g_ * 0.01, n_atoms=n_atoms)

    H = Hamiltonian(capacity=capacity, cavity=cv)

    # print(type(H))
    # print(H.data)
    # print(H.data.data)
    # print(np.shape(H.data.data))
    # H.print_states()
    # H.abs_print("Hamiltonian:", sep='\t\t')

    U = Unitary(H, dt)
    # U.print("Unitary:", sep='\t\t')
    U_conj = U.conj()

    ro_t = Matrix(m=len(ro_0_), n=len(ro_0_), dtype=np.complex128, data=lil_matrix(ro_0_))
    # ro_t.abs_print("DensityMatrix:", sep='\t\t')
    
    # ---------------------------------------------------------------
    L_out = operator_L(ro_t, l_out)
    L_in = operator_L(ro_t, l_in)

    L_op = L_out

    L_type = 'out'
    # ---------------------------------------------------------------
    
    sink = 0

    t = 0

    while True:
        diag_abs = np.abs(ro_t.data.diagonal(), dtype=np.longdouble)

        if L_type == 'out':
            sink_prev = sink

            sink = sink_0 = diag_abs[0]+diag_abs[1]
            sink_1 = diag_abs[2]+diag_abs[3]+diag_abs[4]
            print('sink', ': ', "%.3f" % np.round(sink_0,3), ', ', "%.3f" % np.round(sink_1, 3),'\t| sink_sum: ', "%.3f" % np.round(sink_0+sink_1, 3), ', ro_trace: ', "%.3f" % ro_t.abs_trace(), sep='')
            # print('sink', ': ', np.round(sink_0,3), ', ', np.round(sink_1, 3),', sink_sum: ', np.round(sink_0+sink_1, 3), sep='')
            # print(L_type, ': ', np.round(sink_0,3), ', ', np.round(sink_1, 3),', sink_sum: ', np.round(sink_0+sink_1, 3), sep='')
                
            sink = sink_0
            # print(L_type, sink)
            
            if sink_prev - sink >= sink_precision:
                print(sink, '<', sink_prev)
                exit(0)

            if sink >= 0.99:
                # ro_t = add_ph(ro_t)
 
                # ro_t.abs_print(precision=3, sep='\t')
                # exit(0)
                print('in')
                L_op = L_in
                L_type = 'in'

                sink = sink_1

                ro_t.abs_print(precision=3, sep='\t')
        else:
            sink_prev = sink

            # print(L_type, sink)
            sink_0 = diag_abs[0]+diag_abs[1]
            sink = sink_1 = diag_abs[2]+diag_abs[3]+diag_abs[4]
            print(L_type, ': ', np.round(sink_0,3), ', ', np.round(sink_1, 3),', sink_sum: ', np.round(sink_0+sink_1, 3), sep='')
            
            if sink_prev - sink >= sink_precision:
                print(sink, '<', sink_prev)
                exit(0)

            if sink >= sink_limit:
                # print('out')
                L_op = L_out
                L_type = 'in'

                sink = sink_0

                ro_t.abs_print(precision=3, sep='\t')

                exit(0)

        d = ro_t.data.todense()

        nondiag_is_zeros = True

        for i in range(len(d)):
            for j in range(len(d)):
                if i == j:
                    continue

                ampl = abs(d[i,j])

                if ampl > ampl_precision:
                    nondiag_is_zeros = False
                    break

        if nondiag_is_zeros:
            ampl01.append([abs(ro_t.data[0,0]), abs(ro_t.data[1,1])])
            time_list.append(t)

            # ro_t.print()
            
            print("OK")
            ro_t.abs_print(precision=3, sep='\t')
             
            exit(0)
            break

        ro_t.data = ((U.data).dot(ro_t.data + dt * L_op(ro_t).data)).dot(U_conj.data)
        # ro_t.abs_print(precision=3, sep='\t')

        Assert(abs(1 - ro_t.abs_trace()) <= ro_trace_precision, "ρ(t) is not normed: " + str(ro_t.abs_trace()))

        t += dt

# -------------------------------------
pickle_dump(g_list, path+'/g.pkl')
pickle_dump(time_list, path+'/t.pkl')
# -------------------------------------
pickle_dump(ampl01, path+'/ampl01.pkl')
# -------------------------------------

# =====================================================================================================================
