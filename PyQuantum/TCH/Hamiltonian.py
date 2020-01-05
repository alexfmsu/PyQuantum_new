# =================================================== DESCRIPTION =====================================================
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# =================================================== DESCRIPTION =====================================================



# =================================================== EXAMPLES ========================================================
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# =================================================== EXAMPLES ========================================================



# =====================================================================================================================
# system
from copy import copy
from math import sqrt
# ---------------------------------------------------------------------------------------------------------------------
# scientific
import numpy as np
# import pandas as pd
from scipy.sparse import identity, kron, eye, csc_matrix, bsr_matrix, lil_matrix
# from scipy.sparse import csr_matrix
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.TCH
from PyQuantum.TCH.Basis import *
# from PyQuantum.TCH.Operators.sigma_ij import sigma_ij
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.Tools
from PyQuantum.Tools.Hz import *
from PyQuantum.Tools.Print import *
from PyQuantum.Tools.Sub import *
from PyQuantum.Tools.Matrix import *
# =====================================================================================================================



# =====================================================================================================================
def ab(s, text, cnt):
    coeff = ''

    if cnt == 0:
        return s

    if s != '':
        if cnt >= 1:
            coeff = '+'
        elif cnt == -1:
            coeff = '-'

    if abs(cnt) != 1:
        coeff += str(abs(cnt)) + '*'

    return s + coeff + text
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
def ab_sqrt(s, text, cnt):
    coeff = ''

    if cnt == 0:
        return s

    if s != '':
        if cnt >= 1:
            coeff = '+'
        elif cnt == -1:
            coeff = '-'

    if abs(cnt) != 1:
        coeff += 'sqrt(' + str(abs(cnt)) + ')*'

    return s + coeff + text
# =====================================================================================================================



# =====================================================================================================================
class Hamiltonian(Matrix):
    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- INIT -------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self, args):
        for arg in ['capacity', 'cavity']:
            if arg not in args:
                print(arg + ' not in args')
                exit(0)

        self.capacity = capacity = args['capacity']
        self.cavity = cavity = args['cavity']
        
        if 'RWA' not in args:
            RWA = True
        else:
            if RWA not in [True, False]:
                print('RWA not in [True, False]')
                exit(0)

            RWA = args['RWA']

        basis, basis_list = Basis(capacity, cavity.n_atoms, cavity.n_levels, sink_dim=args['sink_dim'])

        # for i in basis:
        #     print(i)
        # print(len(basis))

        # exit(0)

        self.states = basis
        self.size = len(self.states)
        
        self.states_list = basis_list

        # -------------------------------------------------------------------------------------------------------------
        H0 = self.H0(capacity, cavity.n_atoms, cavity.n_levels, cavity.wc, cavity.wa, cavity.g)
        H1 = self.H1(capacity, cavity.n_atoms, cavity.n_levels, cavity.wc, cavity.wa, cavity.g)

        if RWA:
            HI = self.HI_RWA(capacity, cavity.n_atoms, cavity.n_levels, cavity.wc, cavity.wa, cavity.g)
        else:
            HI = self.HI_EXACT(capacity, cavity.n_atoms, cavity.n_levels, cavity.wc, cavity.wa, cavity.g)
        # -------------------------------------------------------------------------------------------------------------
        


        data = H0 + H1 + HI

        self.get_states_bin()



        # -------------------------------------------------------------------------------------------------------------
        self.H_symb = lil_matrix((self.size, self.size), dtype=str)

        for i in range(self.size):
            for j in range(self.size):
                self.H_symb[i, j] = self.H0_symb[i, j]

                if self.H_symb[i, j] != '':
                    if self.H1_symb[i, j] != '':
                        self.H_symb[i, j] += '+'
                
                self.H_symb[i, j] += self.H1_symb[i, j]
                
                if self.H_symb[i, j] != '':
                    if self.HI_symb[i, j] != '':
                        self.H_symb[i, j] += '+'
                
                self.H_symb[i, j] += self.HI_symb[i, j]
        # -------------------------------------------------------------------------------------------------------------

        # if args['outfile']:
        #     self.to_html(args['outfile'])
        # print(type(data))
        # self.data = csc_matrix(dataa.data, dtype=np.complex128)
        # d = csc_matrix((self.size, self.size))

        super(Hamiltonian, self).__init__(m=self.size, n=self.size, dtype=np.double, data=data)
        # -------------------------------------------------------------------------------------------------------------
        # is_hermitian = np.all(self.data.data.todense().getH()
        #                       == self.data.data.todense())
        # Assert(is_hermitian, 'H is not hermitian')
        # -------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------



    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- STATES -----------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def get_states_bin(self):
        self.states_bin = {}
        
        self.states_sink = {}
        
        for k, v in enumerate(self.states):
            en = v['ph'][0] + sum(v['at'])
            print('en=',en)

            sink = sum(v['sink'])

            if sink not in self.states_sink:
                self.states_sink[sink] = []

            if en not in self.states_bin:
                self.states_bin[en] = []

            self.states_bin[en].append(k)
            self.states_sink[sink].append(k)

    def print_bin_states(self):
        for k, v in self.states_bin.items():
            print(k)

            for k1, v1 in v.items():
                print('\t', k1, ':')

                for i in v1:
                    print('\t\t', self.states[i])
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------


    
    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- H0 ---------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def H0(self, capacity, at_count, n_levels, wc, wa, g):
        H0 = lil_matrix((self.size, self.size))

        self.H0_symb = lil_matrix((self.size, self.size), dtype=str)
        
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                self.H0_symb[i, j] = ''
        
        for i in range(len(self.states)):
            cnt = 0
            
            for k in self.cavity.wc.keys():
                H0[i, i] = self.cavity.wc[k] * self.states[i]['ph'][cnt]

                k_ = k.replace('<->', '')
                # print('k:', k_)
                # exit(0)
                self.H0_symb[i, i] = ab(self.H0_symb[i, i], 'wc'+sub(k_), self.states[i]['ph'][cnt])

                cnt += 1

        # return H0
        return Matrix(m=self.size, n=self.size, dtype=np.float, data=H0)
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------



    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- H1_RWA -----------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def H1(self, capacity, at_count, n_levels, wc, wa, g):
        H1 = lil_matrix((self.size, self.size))
        self.H1_symb = lil_matrix((self.size, self.size), dtype=str)

        for i in range(len(self.states)):
            for j in range(len(self.states)):
                self.H1_symb[i, j] = ''
        
        for i in range(len(self.states)):
            for k in self.cavity.wa.keys():
                for at in range(len(self.cavity.wa[k])):
                    H1[i, i] += self.cavity.wa[k][at] * self.states[i]['at'][at]
                    
                    if k == '0<->1':
                        k_ = k.replace('<->', '')
                        self.H1_symb[i, i] = ab(self.H1_symb[i, i], 'wa'+sub(k_), self.states[i]['at'][at])


        return Matrix(m=self.size, n=self.size, dtype=np.float, data=H1)
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    


    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- HI_RWA -----------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def HI_RWA(self, capacity, at_count, n_levels, wc, wa, g):
        HI = lil_matrix((self.size, self.size))
        self.HI_symb = lil_matrix((self.size, self.size), dtype=str)
        
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                self.HI_symb[i, j] = ''
        
        for i in range(len(self.states)):
            ph_i = self.states[i]['ph']
            at_i = self.states[i]['at']
            sink_i = self.states[i]['sink']

            for j in range(len(self.states)):
                ph_j = self.states[j]['ph']
                at_j = self.states[j]['at']
                sink_j = self.states[j]['sink']
    
                if sink_i != sink_j:
                    continue

                diff_ph = 0
                diff_ph_ji = 0

                for ph in range(len(ph_i)):
                    if ph_i[ph] != ph_j[ph]:
                        if abs(ph_i[ph]) != 1:
                            break

                        diff_ph_ji = ph_j[ph] - ph_i[ph]
                        diff_ph += 1
                        diff_ph_pos = ph
                
                if diff_ph != 1:
                    continue

                for at in range(self.cavity.n_atoms):
                    diff_atoms = 0
                    diff_at_pos = -1
                    diff_at_type = -1
                    
                    if at_i[at] == at_j[at]:
                        continue

                    if abs(at_i[at] - at_j[at]) == 1:
                        diff_atoms_type = max(at_i[at], at_j[at])

                        diff_at_ji = at_j[at] - at_i[at]
                        diff_atoms+=1
                        diff_at_pos = at

                        if diff_atoms > 1:
                            break

                if diff_atoms != 1:
                    continue

                if diff_atoms_type == diff_ph_pos+1 and diff_ph_ji + diff_at_ji == 0:
                    # print(at_i[at], at_j[at])
                    k = max(ph_i[diff_ph_pos], ph_j[diff_ph_pos])
                    HI[i, j] = HI[j, i] = self.cavity.g['0<->1'][diff_at_pos] * sqrt(k)
                    self.HI_symb[i, j] = self.HI_symb[j, i] = ab_sqrt(self.HI_symb[i, j], 'g'+sub('01'), k)
                    # HI_symb[i, j] += ab('g'+sub('0<->1'), k)
                    # print('diff_ph_pos:', diff_ph_pos)
                    # print('diff_atoms_type:', diff_atoms_type)
                    # print(self.states[j], '<->', self.states[i])
                    print()
        
        return Matrix(m=self.size, n=self.size, dtype=np.float, data=HI)
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
        


    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- PRINT STATES -----------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def print_states(self):
        cprint("Basis:\n", "green")

        for k, v in enumerate(self.states):
            print("{:3d}".format(k), ': ', v, sep='')
            # print(k, ': ', v, sep='')

        print()
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------



    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- PRINT ------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    # def print(self):
    #     self.data.print()

    #     print()
        # print(self.data)
        # for i in range(self.size):
        #     for j in range(self.size):
        #         # print(round(self.data[i, j] / self.cavity.wc, 3), end='\t')
        #         print(to_Hz(self.data[i, j]), end='\t')

        #     print()
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------



    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- IPRINT -----------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def iprint(self, filename='H3.html'):
        df = pd.DataFrame()

        data = self.data.data.toarray()

        for i in range(self.size):
            for j in range(self.size):
                if abs(data[i, j] != 0):
                    df.loc[i, j] = to_Hz(abs(data[i, j]))
                else:
                    df.loc[i, j] = ''

        df.index = df.columns = [str(v) for v in self.states_list]

        self.df.to_html(filename)
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------



    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- PRINT_H_SYMB -----------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def print_H_symb(self):
        for i in range(self.size):
            for j in range(self.size):
                print(self.H_symb[i, j], end='\t')
            
            print()
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------



    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- TO_HTML ----------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def to_html(self, filename):
        df = pd.DataFrame(dtype=str)

        data = self.H_symb

        for i in range(self.size):
            for j in range(self.size):
                if abs(data[i, j] != 0):
                    df.loc[i, j] = data[i, j]
                else:
                    df.loc[i, j] = ''

        df.index = df.columns = [str(v) for v in self.states_list]

        self.df = df
        self.df.to_html(filename)        
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
# =====================================================================================================================
