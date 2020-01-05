# =====================================================================================================================
# EXAMPLES:

# ---------------------------------------------------------------------------------------------------------------------
# cv = Cavity(wc=0.2, wa=0.2, g=1, n_atoms=2)
# cv = Cavity(wc=1, wa=1, g=[1.0, 0.5], n_atoms=2)
# cv = Cavity(wc=1, wa=[1, 1], g=[1.0, 0.5], n_atoms=2)
# cv = Cavity(wc=0.2, wa=[0.2, 0.2], g=[1.0, 0.5], n_atoms=2)
# ---------------------------------------------------------------------------------------------------------------------
# =====================================================================================================================



# =====================================================================================================================
# system
import re
import numpy as np
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.Tools
from PyQuantum.Tools.Assert import Assert
from PyQuantum.Tools.Print import cprint
from PyQuantum.Tools.to_Hz import to_Hz
from PyQuantum.Tools.Sub import *
# =====================================================================================================================



# =====================================================================================================================
class Cavity:
    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- INIT -------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self, wc, wa, g, n_atoms, n_levels=2):
        # --------------------------------------------------------
        Assert(isinstance(n_atoms, int), 'n_atoms is not integer')
        Assert(n_atoms > 0, 'n_atoms <= 0')

        self.n_atoms = n_atoms
        # --------------------------------------------------------

        # ---------------------------------------------------------
        Assert(isinstance(n_levels, int), 'n_atoms is not integer')
        Assert(n_levels > 1, 'n_levels <= 2')

        self.n_levels = n_levels

        if self.n_levels == 2:
            # -----------------------
            if isinstance(wc, int):
                wc = float(wc)
            
            if isinstance(wc, float):
                wc = {'0<->1': wc}
            # -----------------------

            # -----------------------
            if isinstance(wa, int):
                wa = float(wa)
            
            if isinstance(wa, float):
                wa = {'0<->1': wa}
            # -----------------------

            # -----------------------
            if isinstance(g, int):
                g = float(g)
            
            if isinstance(g, float):
                g = {'0<->1': g}
            # -----------------------
        # ---------------------------------------------------------
        
        # ---------------------------------------------------------
        if isinstance(wc, dict):
            # Assert(len(wc) == n_levels, 'len(wc) != n_levels')
            
            for k in wc.keys():
                Assert(wc[k] > 0, 'wc <= 0')
        else:
            if isinstance(wc, int):
                wc = float(wc)
            elif not isinstance(wc, float):
                Assert(False, 'wc is not dict or float')

            Assert(wc > 0, 'wc <= 0')
            
        self.wc = wc
        
        self.parse_wc()
        # ---------------------------------------------------------
        
        # ---------------------------------------------------------
        if isinstance(wa, dict):
            # Assert(len(wa) == n_levels-1, 'len(wa) != n_levels')
        
            for k in wa.keys():
                if isinstance(wa[k], int):
                    wa[k] = float(wa[k])

                if isinstance(wa[k], float):
                    wa[k] = [wa[k]] * self.n_atoms
                else:
                    for i in range(len(wa[k])):
                        Assert(wa[k][i] > 0, 'wa <= 0')
        elif isinstance(wa, list):
            Assert(len(wa) == n_atoms, 'len(wa) != n_atoms')

            for k in range(len(wa)):
                Assert(wa[k] > 0, 'wa <= 0')
        else:
            if isinstance(wa, int):
                wa = float(wa)
            elif not isinstance(wa, float):
                Assert(False, 'wa is not list, dict or float')

            Assert(wa > 0, 'wa <= 0')
            

        self.wa = wa
        
        self.parse_wa()
        # ---------------------------------------------------------
        
        # ---------------------------------------------------------
        if isinstance(g, dict):
            # Assert(len(g) == n_levels-1, 'len(g) != n_levels')
        
            for k in g.keys():
                if isinstance(g[k], int):
                    g[k] = float(g[k])

                if isinstance(g[k], float):
                    g[k] = [g[k]] * self.n_atoms
                else:
                    for i in range(len(g[k])):
                        Assert(g[k][i] > 0, 'g <= 0')
        elif isinstance(g, list):
            Assert(len(g) == n_atoms, 'len(g) != n_atoms')
            
            for k in range(len(g)):
                Assert(g[k] > 0, 'g <= 0')
        else:
            if isinstance(g, int):
                g = float(g)
            elif not isinstance(g, float):
                Assert(False, 'g is not list, dict or float')

            Assert(g > 0, 'g <= 0')

        self.g = g
        
        self.parse_g()
        # ---------------------------------------------------------
        
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------


    def parse_wc(self):
        self.wc_parsed = []

        wc_bin = [0] * self.n_levels

        for k in self.wc.keys():
            groups = re.split('<->|-', k)

            # print(k, groups)
            
            groups = [int(i) for i in groups]
            
            Assert(len(groups) == 2, 'len(groups) != 2')
            Assert(groups[0] != groups[1], 'groups[0] == groups[1]')
            Assert(groups[0] >= 0, 'groups[0] < 0')
            Assert(groups[0] <= self.n_levels, 'groups[0] > self.n_levels')
            Assert(groups[1] >= 0, 'groups[1] < 0')
            Assert(groups[1] <= self.n_levels, 'groups[1] > self.n_levels')

            for i in groups:
                wc_bin[i] = 1

            self.wc_parsed.append(groups)

        # print(wc_bin)
        Assert(np.count_nonzero(wc_bin) == self.n_levels, 'np.count_nonzero(wc_bin) != self.n_levels')


    def parse_wa(self):
        self.wa_parsed = []

        wa_bin = [0] * self.n_levels

        for k in self.wa.keys():
            groups = re.split('<->|-', k)

            # print(k, groups)
            
            Assert(len(groups) == 2, 'len(groups) != 2')

            groups = [int(i) for i in groups]

            Assert(groups[0] != groups[1], 'groups[0] == groups[1]')
            Assert(groups[0] >= 0, 'groups[0] < 0')
            Assert(groups[0] <= self.n_levels, 'groups[0] > self.n_levels')
            Assert(groups[1] >= 0, 'groups[1] < 0')
            Assert(groups[1] <= self.n_levels, 'groups[1] > self.n_levels')

            for i in groups:
                wa_bin[i] = 1

            self.wa_parsed.append(groups)

        Assert(np.count_nonzero(wa_bin) == self.n_levels, 'np.count_nonzero(wa_bin) != self.n_levels')

    def parse_g(self):
        self.g_parsed = []

        g_bin = [0] * self.n_levels

        for k in self.g.keys():
            groups = re.split('<->|-', k)

            # print(groups)

            Assert(len(groups) == 2, 'len(groups) != 2')
            
            groups = [int(i) for i in groups]

            Assert(groups[0] != groups[1], 'groups[0] == groups[1]')
            Assert(groups[0] >= 0, 'groups[0] < 0')
            Assert(groups[0] <= self.n_levels, 'groups[0] > self.n_levels')
            Assert(groups[1] >= 0, 'groups[1] < 0')
            Assert(groups[1] <= self.n_levels, 'groups[1] > self.n_levels')

            for i in groups:
                g_bin[i] = 1
        
            self.g_parsed.append(groups)

        Assert(np.count_nonzero(g_bin) == self.n_levels, 'np.count_nonzero(g_bin) != self.n_levels')

    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- WC_INFO ----------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def wc_info(self):
        cprint('wc: ', 'yellow', attrs=['bold'], end='')

        if isinstance(self.wc, dict):
            print()

            for k in self.wc.keys():
                print(k, ':\n\t', to_Hz(self.wc[k]), sep='')
        else:
            print(to_Hz(self.wc), sep='')

        print()
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------



    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- WA_INFO ----------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def wa_info(self):
        cprint('wa: ', 'yellow', attrs=['bold'], end='')

        if isinstance(self.wa, dict):
            print()
            
            for k in self.wa.keys():
                print(k, ':', sep='')
                
                for i in range(len(self.wa[k])):
                    print('\twa', sub(i), ' = ', to_Hz(self.wa[k][i]), sep='')
                
                print()
        elif isinstance(self.wa, list):
            print()

            for i in range(len(self.wa)):
                print('wa', sub(i), ' = ', to_Hz(self.wa[i]), sep='')
            
            print()
        else:
            print(to_Hz(self.wa), sep='')

            print()
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------



    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- G_INFO -----------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def g_info(self):
        cprint('g: ', 'yellow', attrs=['bold'], end='')

        if isinstance(self.g, dict):
            print()

            for k in self.g.keys():
                print(k, ':', sep='')
                
                for i in range(len(self.g[k])):
                    print('\tg', sub(i), ' = ', to_Hz(self.g[k][i]), sep='')

                print()
        elif isinstance(self.g, list):
            print()

            for i in range(len(self.g)):
                print('g', sub(i), ' = ', to_Hz(self.g[i]), sep='')
            
            print()
        else:
            print(to_Hz(self.g), sep='')

            print()
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------



    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- N_ATOMS_INFO -----------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def n_atoms_info(self):
        cprint('n_atoms: ', 'yellow', attrs=['bold'], end='')

        print(self.n_atoms)

        print()
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------



    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- N_LEVELS_INFO ----------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def n_levels_info(self):
        cprint('n_levels: ', 'yellow', attrs=['bold'], end='')

        print(self.n_levels)

        print()
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------



    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- INFO -------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def info(self, title='Cavity:'):
        cprint(title, 'green', attrs=['bold'])

        print()

        self.wc_info()
        self.wa_info()
        self.g_info()
        self.n_atoms_info()
        self.n_levels_info()
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
# =====================================================================================================================



# ======================================================== STUFF ======================================================
# ======================================================== STUFF ======================================================
