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
from scipy.sparse import identity, kron, csc_matrix, lil_matrix
# import pandas as pd
# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.Tools
from PyQuantum.Tools.Matrix import *
# =====================================================================================================================



# =====================================================================================================================
class Hamiltonian(Matrix):
    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- INIT -------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self, capacity, cavity, RWA=True):
        self.capacity = capacity
        self.cavity = cavity

        self.states = [
        	'|00〉',
        	'|01〉',
        	'|10〉',
        	'|11〉',
        	'|0t〉'
        ]

        wc = cavity.wc['0<->1']
        g = cavity.g['0<->1'][0]

        data = [
            #      |00〉      |01〉       |10〉        |11〉      |0t〉
            [		 0, 		0, 		   0, 			0, 		   0], #   |00〉
        	[		 0,        wc, 		   0, 			0, 		   0], #   |01〉
        	[		 0, 		0,        wc, 			0,         g], #   |10〉
        	[		 0, 		0, 		   0,        2*wc, 		   0], #   |11〉
        	[		 0, 		0, 	       g, 			0,        wc]  #   |0t〉
        ]

        self.size = len(data)
        # self.data = Matrix(m=self.size, n=self.size, dtype=np.complex128, data=data)
        
        super(Hamiltonian, self).__init__(m=self.size, n=self.size, dtype=np.complex128, data=lil_matrix(data))
        #     # m=self.size, n=self.size, dtype=np.float)
        self.check_hermiticity()
        
        # print(self.data.data)
        # self.data = 
        # self.print()
        # print(self.m)
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    

    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- PRINT STATES -----------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def print_states(self):
        cprint("Base states:", "green")

        for i in self.states:
            print(i)

        print()
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------


    
    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- PRINT ------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    # def print(self):
    # 	for i in self.data.toarray():
    # 		print(i)
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
# =====================================================================================================================