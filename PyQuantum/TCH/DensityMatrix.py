# =================================================== DESCRIPTION =====================================================
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# =================================================== DESCRIPTION =====================================================



# =================================================== EXAMPLES ========================================================
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# =================================================== EXAMPLES ========================================================



# =====================================================================================================================

# scientific
import numpy as np
# import pandas as pd
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.TCH
from PyQuantum.TCH.WaveFunction import WaveFunction
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.Tools
from PyQuantum.Tools.Print import print
from PyQuantum.Tools.Assert import Assert
from PyQuantum.Tools.Matrix import Matrix
# =====================================================================================================================
from scipy.sparse import lil_matrix



# =====================================================================================================================
class DensityMatrix(Matrix):
    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- INIT -------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self, wf):
        if isinstance(wf, WaveFunction):
            wf_data = wf.data
            ro_data = wf_data.dot(wf_data.getH())
            
            self.states = wf.states
        elif isinstance(wf, lil_matrix):
            ro_data = wf

            self.sink_base = None
        
        # Assert(isinstance(wf, WaveFunction), "wf is not WaveFunction")

        # super(DensityMatrix, self).__init__(
        #     m=wf.m, n=wf.m, dtype=np.complex128, data=None)

        # wf_data = wf.data

        # print('t2:',type(wf_data))
        # print('t2:',type(wf_data.getH()))
        # ro_data = lil_matrix(ro_data)
        # Assert(np.shape(ro_data) == (self.m, self.n), "size mismatch")

        # print('t:', type(ro_data))
        # exit(0)
        # self.data = ro_data
        # self.data = np.matrix(ro_data, dtype=np.complex128)

        self.size = np.shape(ro_data)[0]

        super(DensityMatrix, self).__init__(m=self.size, n=self.size, dtype=np.complex128, data=ro_data)

        # exit(0)
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------



    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- NORMALIZE --------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def renormalize(self):
        self.data = (self.data+self.data.conj())/2.0
        self.data /= self.trace_abs()
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

    def print_sink(self, energy=None, precision=3, sep=', '):
        print_format = "%." + str(precision) + "f"
        
        if energy is None:    
            s = str.join(sep, [print_format % np.round(self.sink[i], precision) for i in self.sink.keys()])
            
            print(s)
        else:
            Assert(str(energy) in self.sink.keys(), "str(energy) not in self.sink.keys()")
            
            print(print_format % np.round(self.sink[str(energy)], precision))


    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- ENERGY -----------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def energy(self, capacity, n_atoms, states_bin, diag_abs):
        # energy = [0] * (capacity['0_1'] + n_atoms + 1)
        energy = dict.fromkeys(states_bin.keys())

        # energy = [0] * (capacity + n_atoms+1)

        # for i in states_bin:
        #     print(i)
        # print(energy)
        for i in states_bin.keys():
            # print(i)
            # print(states_bin[i])
            # print(np.sum(diag_abs[states_bin[i]]))
            energy[i] = np.sum(diag_abs[states_bin[i]])
            # for j in states_bin[i]:
            #     energy[i] += diag_abs[j]
            # energy[i] *= i

        # for i in range(1, len(states_bin)):
        # exit(0)
        return energy
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
# =====================================================================================================================

    def trace_abs(self):
        return np.sum(self.diag_abs())

    def diag_abs(self):
        return np.abs(self.data.diagonal(), dtype=np.longdouble)

    def set_sink_base(self, sink_base):
        self.sink_base = sink_base

        self.sink = dict.fromkeys(sink_base)

    def get_sink(self):
        diag_abs = self.diag_abs()

        for k, v in self.sink_base.items():
            self.sink[k] = np.sum(diag_abs[v])

        return self.sink

# ======================================================== STUFF ======================================================
# def iprint(self):
#     df = pd.DataFrame()

#     for i in range(self.size):
#         for j in range(self.size):
#             df.loc[i, j] = self.data[i, j]

#     df.index = df.columns = [str(v) for v in self.states.values()]

#     self.df = df
# ======================================================== STUFF ======================================================
