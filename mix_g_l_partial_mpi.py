# =====================================================================================================================
# system
from math import sqrt
import copy
import os
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
from PyQuantum.Tools.Print import print
from PyQuantum.Tools.LoadPackage import *
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.Constants
import PyQuantum.Constants as Constants
# =====================================================================================================================
# import config as cfg
# =====================================================================================================================
from PyQuantum.Tools.MPI.MPI import *
from PyQuantum.Tools.MPI.ParallelFor import *

mpirank = MPI_Comm_rank()
mpisize = MPI_Comm_size()
# =====================================================================================================================
from shutil import copyfile
# =====================================================================================================================

rootdir = os.getcwd()

# ===============================================
MPI_Barrier()

if mpirank == 0:
    # build_config = os.system('perl mix_config.pl')

    # MPI_Assert(build_config == 0, 'Build config: failed', rootdir, err_file='slurm.err')

    # print('Build config... ', end='')
    # print('ok')
    config_filename = sys.argv[1]

    cfg = load_pkg(config_filename, config_filename)

    # print('Build config... ', end='')
    # print('ok')

    mkdir(cfg.path)
    copyfile('config.py', cfg.path+'/'+'config.py')

MPI_Barrier()
# config_filename = sys.argv[1]
# print(config_filename)
# exit(0)
cfg = load_pkg(config_filename, config_filename)


    # print('dt=', time_unit_full(cfg.dt))
    # print('dt=', to_Hz(cfg.l_0))
    
    # info = {
    #     # ---------------------------------------
    #     'capacity': capacity,
    #     # ---------------------------------------
    #     'n_atoms': n_atoms,
    #     # ---------------------------------------
    #     'wc': wc,
    #     'wc(Hz)': to_Hz(wc),
    #     'wa': wa,
    #     'wa(Hz)': to_Hz(wa),
    #     # ---------------------------------------
    #     'g_0': g_0,
    #     'g_step': g_step,
    #     'g_1': g_1,
    #     # ---------------------------------------
    #     'l_0': l_0,
    #     'l_step': l_step,
    #     'l_1': l_1,
    #     # 'l': l,
    #     # 'l(Hz)': to_Hz(l),
    #     # 'l/wc': l/wc,
    #     # ---------------------------------------
    #     'dt': dt,
    #     'dt(s)': time_unit_full(dt),
    #     # 'dt*l': dt*l,
    #     # ---------------------------------------
    #     'sink_limit': sink_limit,
    #     # ---------------------------------------
    #     'sink_precision': sink_precision,
    #     # 'ro_trace_precision': ro_trace_precision,
    #     'ampl_precision': ampl_precision,
    #     # ---------------------------------------
    #     'path': path,
    #     # ---------------------------------------
    # }

    # pickle_dump(info, path+'/info.pkl')

# ===========================================
# MPI_Abort('exit(0)')
# path = 'out/mix'

capacity = cfg.capacity

n_atoms = cfg.n_atoms

wc = cfg.wc
wa = cfg.wa

g_0 = cfg.g_0
g_1 = cfg.g_1
g_step = cfg.g_step

l_0 = cfg.l_0
l_1 = cfg.l_1
l_step = cfg.l_step

dt = cfg.dt

sink_limit = cfg.sink_limit

sink_precision = cfg.sink_precision
# ro_trace_precision = cfg.ro_trace_precision
ampl_precision = cfg.ampl_precision

path = cfg.path

outfile  = ''
outfile  = 'l' + str(l_0) + '_' + str(l_1) + '_' + str(l_step) + '_'
outfile += 'g' + str(g_0) + '_' + str(g_1) + '_' + str(g_step) + '_'
outfile += 'dt' + str(time_unit_full(dt)).replace(' ','') + '_'
outfile += 'sink_limit' + str(sink_limit)

path += '/' + outfile

node_print('\n', 0)
node_print('Path: ' + path, 0)
node_print('Outfile: ' + outfile, 0)
# ===============================================

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
    # 'l': Constants.wc
}

def add_ph(ro_t):
    for i in range(0, 2):
        for j in range(0, 2):
            ro_t.data[i+2, j+2], ro_t.data[i, j] = ro_t.data[i, j], ro_t.data[i+2, j+2]

    return ro_t
# END----------------------------------------------------- LINDBLAD ---------------------------------------------------



g_arange = np.round(np.arange(g_0, g_1+g_step/2, g_step), 3)

n1, n2 = n_batches(len(g_arange))

# out_g_l = []
# out_time_l = []
# out_n_clicks_l = []

# print('node: ', mpirank, ', n1=', n1, ', n2=', n2, sep='')
# ---------------------------------------



l_list = np.arange(l_0, l_1+l_step/2, l_step)
l_list = np.round(l_list, 3)

def node_evolution():
    # print(l_list)

    MPI_Assert(len(l_list) == mpisize, 'len(l_list) = ' + str(len(l_list)) +', len(l_list) == mpisize; mpisize = ' + str(mpisize) + ", l_list:" + str.join(',', [str(t) for t in l_list]), rootdir, outfile+'.err')

    l_coeff = l_list[mpirank]
    l_out['l'] = Constants.wc * l_coeff
    

    for g_coeff in g_arange:
        # print(g_coeff)
        out_g_l = []
        out_time_l = []
        out_n_clicks_l = []
        # pickle_dump(out_g_l, 'g_'+str(g_coeff)+'.pkl')
        # pickle_dump(out_time_l, 't'+str(g_coeff)+'.pkl')
        # pickle_dump(out_n_clicks_l, 'n_clicks'+str(g_coeff)+'.pkl')
        # break

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

            # ro_t.print_sink()
                
            if L_type == 'out':
                if sink['1'] >= sink_limit:
                    # print('node ', mpirank, ': ', 'yes', flush=True)
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

                    # print('t:', t)
                    # print('in:')
                    # ro_t.abs_print(precision=3, sep='\t')
                    # print('-'*50)
                    # print('-'*50)
                    # print()
                    # ===================================
            
            ro_t.data = ((U.data).dot(ro_t.data + dt * L_op(ro_t).data)).dot(U_conj.data)
            ro_t.renormalize()

            t += dt
        # --------------------------------------------------------------------
        # node
        pickle_dump(out_g_l, 'g_'+str(g_coeff)+'.pkl')
        pickle_dump(out_time_l, 't'+str(g_coeff)+'.pkl')
        pickle_dump(out_n_clicks_l, 'n_clicks'+str(g_coeff)+'.pkl')
        # break
        # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # root
    # gather_file(path, 'g', range(mpisize))
    # gather_file(path, 't', range(mpisize))
    # gather_file(path, 'n_clicks', range(mpisize))

    node_print("\nOK", 0, filename=outfile+'.out', to_print=True)
    # --------------------------------------------------------------------

# =====================================================================================================================

# print(l_list)
parallel_for(func=node_evolution, path=path, prefix='l_', var=l_list)
