# ---------------------------------------------------------------------------------------------------------------------
# system
from copy import copy
# ---------------------------------------------------------------------------------------------------------------------
# scientific
import numpy as np
import pandas as pd
from scipy.sparse import identity, kron, eye, csc_matrix
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.Tools
from PyQuantum.Tools.Hz import *
from PyQuantum.Tools.Print import *
from PyQuantum.Tools.Sub import *
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.Common
from PyQuantum.Common.Matrix import *
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.TC
# from PyQuantum.TC.AtomicBase import *
from PyQuantum.TCH.Basis import *
# from PyQuantum.TC.FullBase import *

from PyQuantum.TCH.Operators.sigma_ij import sigma_ij
# ---------------------------------------------------------------------------------------------------------------------



class Hamiltonian(Matrix):
    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- INIT -------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self, capacity, cavity, RWA=True, iprint=False, iprint_symb=False, sink_dim=[1,1]):
        self.capacity = capacity
        self.cavity = cavity

        basis = Basis(capacity, cavity.n_atoms, cavity.n_levels)

        self.states = basis.basis

        # -------------------------------------------------------------------------------------------------------------
        self.H0 = self.H0(
            capacity, cavity.n_atoms, cavity.n_levels, cavity.wc, cavity.wa, cavity.g)
        self.H1 = self.H1(
            capacity, cavity.n_atoms, cavity.n_levels, cavity.wc, cavity.wa, cavity.g)

        if RWA:
            self.HI = self.HI_RWA(
                capacity, cavity.n_atoms, cavity.n_levels, cavity.wc, cavity.wa, cavity.g)
        else:
            self.HI = self.get_Hint_EXACT(
                capacity, cavity.n_atoms, cavity.n_levels, cavity.wc, cavity.wa, cavity.g)

        Assert(np.shape(self.H0) == np.shape(self.H1), "size mismatch")
        Assert(np.shape(self.H1) == np.shape(self.HI), "size mismatch")

        # self.print_states()
        # self.data = self.H0
        # self.data = self.H1
        # self.data = self.HI
        self.data = self.H0 + self.H1 + self.HI
        # print(np.shape(self.data)[0])
        # self.size = np.shape(self.data)[0]
        self.data = kron(self.data, eye(2))
        self.data = kron(self.data, eye(2))
        # print(np.shape(self.data)[0])

        self.cut_states(capacity)

        self.print_states()

        self.size = np.shape(self.data)[0]

        print('size:', self.size)
        print('len(states):', len(self.states))
        Assert(self.size == len(self.states), "size mismatch")

        # print(kron(self.states, [0, 1]))
        self.get_states_bin()

        # ----------------------------------
        # cprint('H0:', color="green")
        # print(self.H0.toarray())
        # cprint('H1:', color="green")
        # print(self.H1.toarray())
        # cprint('HI:', color="green")
        # print(self.HI.toarray())
        # cprint('H:', color="green")
        # print(self.data.toarray())

        # if not self.data.is_hermitian():
        # exit(0)
        is_hermitian = np.all(self.data.todense().getH()
                              == self.data.todense())
        Assert(is_hermitian, 'H is not hermitian')

        if iprint_symb:
            self.iprint_symb('H3_symb.html')

        # if iprint:
        self.iprint('H3.html')
        # exit(0)
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------



    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- STATES -----------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def cut_states(self, capacity):
        to_rm = []

        for i in range(len(self.states)):
            en_1 = self.states[i][0] + \
                np.count_nonzero(np.array(self.states[i][2]) >= 1) + self.states[i][3][0]
            en_2 = self.states[i][1] + \
                np.count_nonzero(np.array(self.states[i][2]) == 2) + self.states[i][3][1]

            if en_1 != capacity['0_1'] or en_2 != capacity['1_2']:
                to_rm.append(i)

        self.data = self.data.toarray()

        for i in to_rm[::-1]:
            self.data = np.delete(self.data, i, axis=0)
            self.data = np.delete(self.data, i, axis=1)

            del self.states[i]

        self.data = csc_matrix(self.data)

    def get_states_bin(self):
        states_bin = {
            '00': [],
            '01': [],
            '10': [],
            '11': [],
        }

        for k, v in enumerate(self.states):
            if v[3][0] == 0 and v[3][1] == 0:
                states_bin['00'].append(k)
            elif v[3][0] == 0 and v[3][1] == 1:
                states_bin['01'].append(k)
            elif v[3][0] == 1 and v[3][1] == 0:
                states_bin['10'].append(k)
            elif v[3][0] == 1 and v[3][1] == 1:
                states_bin['11'].append(k)

        # en = {}

        # for k, v in enumerate(self.states):
        #     en['0_1'] = v[0] + np.count_nonzero(np.array(v[2]) >= 1)
        #     en['1_2'] = v[1] + np.count_nonzero(np.array(v[2]) == 2)

        #     if en['0_1'] not in states_bin['0_1']:
        #         states_bin['0_1'][en['0_1']] = []
        #     if en['1_2'] not in states_bin['1_2']:
        #         states_bin['1_2'][en['1_2']] = []

        #     states_bin['0_1'][en['0_1']].append(k)
        #     states_bin['1_2'][en['1_2']].append(k)

        self.states_bin = states_bin

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
        adiag = {}
        across = {}
        a = {}
        acrossa = {}

        # ------------------------------------------------------------------------------------------------------------------
        for k in capacity.keys():
            adiag[k] = np.sqrt(np.arange(1, capacity[k]+1))
            across[k] = np.diagflat(adiag[k], -1)
            a[k] = np.diagflat(adiag[k], 1)
            acrossa[k] = np.dot(across[k], a[k])
        # ------------------------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------------------------
        at_dim = pow(n_levels, at_count)

        I_at = identity(at_dim)
        I_ph = {}

        for k in capacity.keys():
            I_ph[k] = identity(capacity[k]+1)
        # ------------------------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------------------------
        H_field = {}
        self.H_field_symb = {}

        H_field['0_1'] = kron(acrossa['0_1'], I_ph['1_2'])
        H_field['0_1'] = kron(H_field['0_1'], I_at)

        H_field['1_2'] = kron(I_ph['0_1'], acrossa['1_2'])
        H_field['1_2'] = kron(H_field['1_2'], I_at)

        H_dim = (capacity['0_1']+1) * (capacity['1_2']+1) * \
            pow(n_levels, at_count)

        H0 = csc_matrix((H_dim, H_dim))

        for k in capacity.keys():
            H0 += wc[k] * H_field[k]

        return H0
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------



    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- H1_RWA -----------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def H1(self, capacity, at_count, n_levels, wc, wa, g):
        # -----------------------------------------
        sigmadiag = range(1, n_levels)
        sigmacross = np.diagflat(sigmadiag, -1)
        sigma = np.diagflat(sigmadiag, 1)
        sigmacrosssigma = np.dot(sigmacross, sigma)
        # -----------------------------------------
        I_ph = {}

        for k in capacity.keys():
            I_ph[k] = identity(capacity[k]+1)
        # ------------------------------------------------------------------------------------------------------------------
        
        H_ph = I_ph['0_1']
        H_dim = (capacity['0_1']+1)

        for k in capacity.keys()[1:]:
            H_dim *= (capacity[k]+1)
            H_ph = kron(H_ph, I_ph[k])
        
        H_dim *= pow(n_levels, at_count)
        # H_dim = (capacity['0_1']+1) * (capacity['1_2']+1) * \
            # pow(n_levels, at_count)

        H1 = csc_matrix((H_dim, H_dim))
        # ------------------------------------------------------------------------------------------------------------------
        self.H_atoms_symb = copy(H1)

        for i in range(1, at_count+1):
            elem = sigmacrosssigma

            at_prev = identity(pow(n_levels, i-1))
            elem = kron(at_prev, elem)

            at_next = identity(pow(n_levels, at_count-i))
            elem = kron(elem, at_next)

            H1 += wa['0_1'][i-1] * kron(H_ph, elem)

            self.H_atoms_symb += kron(H_ph, elem)

        return H1
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------



    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- HI_RWA -----------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def HI_RWA(self, capacity, at_count, n_levels, wc, wa, g):
        adiag = {}
        across = {}
        a = {}
        acrossa = {}
        sigma = {}
        sigmacross = {}
        sigmacrosssigma = {}
        sigmadiag = {}

        # ------------------------------------------------------------------------------------------------------------------
        sigma['0_1'] = sigma_ij(0, 1, n_levels=3)
        sigma['1_2'] = sigma_ij(1, 2, n_levels=3)

        for k in capacity.keys():
            adiag[k] = np.sqrt(np.arange(1, capacity[k]+1))
            across[k] = np.diagflat(adiag[k], -1)
            a[k] = np.diagflat(adiag[k], 1)
            acrossa[k] = np.dot(across[k], a[k])

            sigmacross[k] = np.transpose(sigma[k])
            sigmacrosssigma[k] = np.dot(sigmacross[k], sigma[k])
        # ------------------------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------
        H_dim = (capacity['0_1']+1) * (capacity['1_2']+1) * \
            pow(n_levels, at_count)

        # HI = bsr_matrix((H_dim, H_dim))

        I_ph = {}
        for k in capacity.keys():
            I_ph[k] = identity(capacity[k]+1)

        H_int = {}

        for k in capacity.keys():
            H_int[k] = csc_matrix((H_dim, H_dim))
        # ------------------------------------------------------------------------------------------------------------------
        self.H_int_symb = np.zeros([H_dim, H_dim])

        H_int['0_1'] = None
        H_int['1_2'] = None

        for i in range(1, at_count+1):
            # ------------------------------------------------
            elem = kron(across['0_1'], I_ph['1_2'])

            before = identity(pow(n_levels, i-1))
            elem = kron(elem, before)

            elem = kron(elem, sigma['0_1'])

            after = identity(pow(n_levels, at_count-i))
            elem = kron(elem, after)
            # exit(0)
            if H_int['0_1'] is None:
                H_int['0_1'] = g['0_1'] * elem
            else:
                H_int['0_1'] += g['0_1'] * elem

            self.H_int_symb += elem
            # ------------------------------------------------
            elem = kron(a['0_1'], I_ph['1_2'])

            before = identity(pow(n_levels, i-1))
            elem = kron(elem, before)

            elem = kron(elem, sigmacross['0_1'])

            after = identity(pow(n_levels, at_count-i))
            elem = kron(elem, after)

            H_int['0_1'] += g['0_1'] * elem
            self.H_int_symb += elem

        for i in range(1, at_count+1):
            # ------------------------------------------------
            elem = kron(I_ph['0_1'], across['1_2'])

            before = identity(pow(n_levels, i-1))
            elem = kron(elem, before)

            elem = kron(elem, sigma['1_2'])

            after = identity(pow(n_levels, at_count-i))
            elem = kron(elem, after)

            if H_int['1_2'] is None:
                H_int['1_2'] = g['1_2'] * elem
            else:
                H_int['1_2'] += g['1_2'] * elem

            self.H_int_symb += elem
            # ------------------------------------------------
            elem = kron(I_ph['0_1'], a['1_2'])

            before = identity(pow(n_levels, i-1))
            elem = kron(elem, before)

            elem = kron(elem, sigmacross['1_2'])

            after = identity(pow(n_levels, at_count-i))
            elem = kron(elem, after)

            H_int['1_2'] += g['1_2'] * elem
            self.H_int_symb += elem

        HI = H_int['0_1'] + H_int['1_2']

        return HI
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
    def print(self):
        for i in range(self.size):
            for j in range(self.size):
                print(round(self.data[i, j] / self.cavity.wc, 3), end='\t')
                # print(to_Hz(self.data[i, j]), end='\t')

            print()
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------



    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- IPRINT -----------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def iprint(self, filename='H3.html'):
        df = pd.DataFrame()

        data = self.data.toarray()

        for i in range(self.size):
            for j in range(self.size):
                if abs(data[i, j] != 0):
                    df.loc[i, j] = to_Hz(abs(data[i, j]))
                else:
                    df.loc[i, j] = ''

        df.index = df.columns = [str(v) for v in self.states]

        self.df = df
        self.df.to_html(filename)
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------



    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- IPRINT_SYMB ------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def iprint_symb(self, filename):
        df = pd.DataFrame()

        data_01 = self.H_field_symb['0_1'].toarray()
        data_12 = self.H_field_symb['1_2'].toarray()

        data_atoms = self.H_atoms_symb.toarray()
        data_int = self.H_int_symb
        # data_atoms = self.H_atoms_symb.toarray()

        for i in range(self.size):
            for j in range(self.size):
                df.loc[i, j] = ''

        for i in range(self.size):
            for j in range(self.size):
                # ---------------------------------------------------------------------------------
                # WC_01
                if data_01[i, j] == 0:
                    pass
                elif data_01[i, j] == 1:
                    df.loc[i, j] += str('wc')+sub('01')
                else:
                    df.loc[i, j] += str('wc')+sub('01') + \
                        '*' + str(data_01[i, j])
                # ---------------------------------------------------------------------------------

                # ---------------------------------------------------------------------------------
                # WC_12
                if data_12[i, j] == 0:
                    pass
                elif data_12[i, j] == 1:
                    if df.loc[i, j] != '':
                        df.loc[i, j] += '+'

                    df.loc[i, j] += str('wc')+sub('02')
                else:
                    if df.loc[i, j] != '':
                        df.loc[i, j] += '+'

                    df.loc[i, j] += str('wc')+sub('12') + \
                        '*' + str(data_12[i, j])
                # ---------------------------------------------------------------------------------

                # ---------------------------------------------------------------------------------
                # atoms
                # print(data_atoms)
                if data_atoms[i, j] == 0:
                    pass
                elif data_atoms[i, j] == 1:
                    if df.loc[i, j] != '':
                        df.loc[i, j] += '+'

                    df.loc[i, j] += str('wa')+sub('02')
                else:
                    if df.loc[i, j] != '':
                        df.loc[i, j] += '+'

                    df.loc[i, j] += str('wa')+sub('12') + \
                        '*' + str(data_atoms[i, j])
                # ---------------------------------------------------------------------------------

                # ---------------------------------------------------------------------------------
                if data_int[i, j] == 0:
                    pass
                else:
                    # if df.loc[i, j] != '':
                        # df.loc[i, j] += '+'

                    df.loc[i, j] += str('g')+sub('12') + \
                        '*' + str(data_int[i, j])
                # ---------------------------------------------------------------------------------
        df.index = df.columns = [str(v) for v in self.states]

        self.df = df
        self.df.to_html(filename)
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

# =====================================================================================================================
