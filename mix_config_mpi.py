# =====================================================================================================================
# system
from math import sqrt
import copy
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.TCH
from PyQuantum.TCH.Cavity import Cavity
from Mix.Hamiltonian import Hamiltonian

from PyQuantum.TCH.Unitary import Unitary
from PyQuantum.TCH.DensityMatrix import DensityMatrix
from PyQuantum.TCH.Lindblad import operator_a, operator_L
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.Tools
from PyQuantum.Tools.Matrix import * # ?
from PyQuantum.Tools.Pickle import *
from PyQuantum.Tools.Mkdir import *
from PyQuantum.Tools.to_Hz import *
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.Constants
import PyQuantum.Constants as Constants
# =====================================================================================================================
import config as cfg
# =====================================================================================================================
from PyQuantum.Tools.MPI import *

mpirank = MPI_Comm_rank()
mpisize = MPI_Comm_size()
# =====================================================================================================================



# =====================================================================================================================
# ===========================================
capacity = cfg.capacity

n_atoms = cfg.n_atoms

wc = cfg.wc
wa = cfg.wa

g_0 = cfg.g_0
g_1 = cfg.g_1
g_step = cfg.g_step

l = cfg.l

dt = cfg.dt

sink_limit = cfg.sink_limit

sink_precision = cfg.sink_precision
ro_trace_precision = cfg.ro_trace_precision
ampl_precision = cfg.ampl_precision

path = 'out/mix'

path_l_wc = wc/l
# if path_l_wc != int(path_l_wc):
#     print('path_l_wc != int(path_l_wc)')
#     exit(0)
# path_l_wc = int(path_l_wc)

outfile = 'l_wc' + str(path_l_wc)

path_dt_l = dt*l

outfile += '_dt_l_' + str(path_dt_l)

path += '/' + outfile

# node_print('path:' + path, 0)
# ===============================================

# ===============================================
MPI_Barrier()

if mpirank == 0:
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

MPI_Barrier()
# ===============================================


# =====================================================================================================================
# alpha = complex(1.0 / sqrt(3), 0)
# beta = complex(sqrt(2) / sqrt(3), 0)

# BEGIN--------------------------------------------------- DENSITY MATRIX ---------------------------------------------
alpha = complex(1.0 / sqrt(2.0), 0)
beta = complex(1.0 / sqrt(2.0), 0)

ro_0_ = [
#          |00〉           |0s〉                   |10〉                        |1s〉          |0t〉
#   |0〉|00〉|1〉    |0〉|s〉|1〉            |1〉|00〉|0〉                 |1〉|s〉|0〉    |0〉|t〉|0〉
    [        0,             0,                       0,                          0,             0],
    [        0,             0,                       0,                          0,             0],
    [        0,             0,           abs(alpha)**2,     alpha*beta.conjugate(),             0],
    [        0,             0,  beta*alpha.conjugate(),               abs(beta)**2,             0],
    [        0,             0,                       0,                          0,             0]
]
ro_0 = DensityMatrix(lil_matrix(ro_0_))
ro_0.set_sink_base({
        '0': [2, 3, 4],
        '1': [0, 1],
    }
)
# END----------------------------------------------------- DENSITY MATRIX ---------------------------------------------

# BEGIN--------------------------------------------------- LINDBLAD ---------------------------------------------------
_a = [
#          |00〉           |0s〉                   |10〉                        |1s〉          |0t〉
#   |0〉|00〉|1〉    |0〉|s〉|1〉            |1〉|00〉|0〉                 |1〉|s〉|0〉    |0〉|t〉|0〉
    [        0,             0,                       1,                          0,          0],
    [        0,             0,                       0,                          1,          0],
    [        0,             0,                       0,                          0,          0],
    [        0,             0,                       0,                          0,          0],
    [        0,             0,                       0,                          0,          0]
]
a = Matrix(m=len(_a), n=len(_a), dtype=np.complex128, data=lil_matrix(_a))

l_out = {
    'L': a,
    'l': l
}

def add_ph(ro_t):
    for i in range(0, 2):
        for j in range(0, 2):
            ro_t.data[i+2, j+2], ro_t.data[i, j] = ro_t.data[i, j], ro_t.data[i+2, j+2]

    return ro_t
# END----------------------------------------------------- LINDBLAD ---------------------------------------------------



g_arange = np.round(np.arange(g_0, g_1+g_step/2, g_step), 3)

n1, n2 = n_batches(len(g_arange))

out_g_l = []
out_time_l = []
out_n_clicks_l = []

# print(g_arange)
# print('node: ', mpirank, ', n1=', n1, ', n2=', n2, sep='')

def parallel_for(start, end, step):
    for g_coeff in g_arange[n1:n2]:

for l_coeff in range(100):
for g_coeff in g_arange[n1:n2]:
    # print(mpirank,': g_coeff:', g_coeff)
    # print('g_coeff:', g_coeff)
    # out_g_l.append(g_coeff)
    # continue

    cv = Cavity(wc=wc, wa=wa, g=Constants.wc * g_coeff * 0.01, n_atoms=n_atoms)

    H = Hamiltonian(capacity=capacity, cavity=cv)

    U = Unitary(H, dt)
    U_conj = U.conj()

    ro_t = copy.deepcopy(ro_0)
    # ---------------------------------------------------------------
    L_out = operator_L(ro_t, l_out)
    
    L_op = L_out

    L_type = 'out'
    # ---------------------------------------------------------------
    
    sink = ro_t.get_sink()
    sink_prev = copy.deepcopy(sink)

    n_clicks = 0

    t = 0

    while True:
        sink_prev = copy.deepcopy(sink)
        sink = ro_t.get_sink()

        ro_t.print_sink()
        # print(np.round(sink['0'],3), ', ', np.round(sink['1'], 3),', sink_sum: ', np.round(sink['0']+sink['1'], 3), sep='')
        # print(L_type, ': ', np.round(sink['0'],3), ', ', np.round(sink['1'], 3),', sink_sum: ', np.round(sink['0']+sink['1'], 3), sep='')
            
        if L_type == 'out':
            # if sink['0'] > sink_prev['0']:
            #     err_msg = "sink['0'] > sink_prev['0']: "
            #     err_msg += str(sink['0']) + ' > ' + str(sink_prev['0']) + '\n'

            #     MPI_Abort(error=err_msg, filename=outfile+'.err', to_print=True)
            # if sink['0'] - sink_prev['0'] >= sink_precision:
                # err_msg = "sink['0'] - sink_prev['0'] >= sink_precision: "
                # err_msg += str(sink['0']) + ' - ' + str(sink_prev['0']) + ' >= ' + str(sink_precision)
                
            
            # if sink_prev['1'] > sink['1']:
            #     err_msg = "sink_prev['1'] > sink['1']: "
            #     err_msg += str(sink_prev['1']) + ' > ' + str(sink['1']) + '\n'

            #     MPI_Abort(error=err_msg, filename=outfile+'.err', to_print=True)
            # if sink_prev['1'] - sink['1'] >= sink_precision:
                # err_msg = "sink_prev['1'] - sink['1'] >= sink_precision: "
                # err_msg += str(sink_prev['1']) + ' - ' + str(sink['1']) + ' >= ' + str(sink_precision)
                


            if sink['1'] >= sink_limit:
                # print('node ', mpirank, ': ', 'yes')
                n_clicks += 1
                # ===========================================
                d = ro_t.data.todense()

                nndiag_is_zeros = True

                for i in range(4):
                    for j in range(4):
                        if i == j and i >= 0 and i <= 3:
                            continue

                        ampl = abs(d[i,j])
                        
                        if ampl > ampl_precision:
                            nndiag_is_zeros = False
                            break

                    if not nndiag_is_zeros:
                        break

                # print('nndiag_is_zeros:', nndiag_is_zeros)
                
                if nndiag_is_zeros:
                    out_g_l.append(g_coeff)
                    out_time_l.append(t)
                    out_n_clicks_l.append(n_clicks)
                    # print("OK")
                    # ro_t.abs_print(precision=3, sep='\t')
                    
                    break
                # ===========================================
                
                # ===================================
                # print('-'*50)
                # print('-'*50)
                # print('out:')
                # ro_t.abs_print(precision=3, sep='\t')
                # print('-'*50)
                
                ro_t = add_ph(ro_t)
                
                sink = ro_t.get_sink()

                # print('sink:', sink)
                # t+=dt
                # continue 
                # L_type = 'in'
                # sink = 0
                # print('t:', t)
                # print('in:')
                # ro_t.abs_print(precision=3, sep='\t')
                # print('-'*50)
                # print('-'*50)
                # print()
                # ===================================
        
        ro_t.data = ((U.data).dot(ro_t.data + dt * L_op(ro_t).data)).dot(U_conj.data)
        ro_t.renormalize()
        # ro_t.abs_print(precision=3, sep='\t')

        # if abs(1 - ro_t.abs_trace()) > ro_trace_precision:
            # MPI_Abort(error="ρ(t) is not normed: " + str(ro_t.abs_trace()) + '\n', filename=outfile+'.err', to_print=True)

        t += dt

# --------------------------------------------------------------------
# node
pickle_dump(out_g_l, path+'/g_' + str(mpirank) + '.pkl')
pickle_dump(out_time_l, path+'/t_' + str(mpirank) + '.pkl')
pickle_dump(out_n_clicks_l, path+'/n_clicks_' + str(mpirank) + '.pkl')
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# root
gather_file(path, 'g', range(mpisize))
gather_file(path, 't', range(mpisize))
gather_file(path, 'n_clicks', range(mpisize))

node_print("OK", 0, filename=outfile+'.out', to_print=True)
# --------------------------------------------------------------------
# =====================================================================================================================


