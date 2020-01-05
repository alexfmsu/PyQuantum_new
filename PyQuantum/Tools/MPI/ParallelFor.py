# =====================================================================================================================
# system
import os
# =====================================================================================================================
# PyQuantum.Tools
from PyQuantum.Tools.Mkdir import *
from PyQuantum.Tools.MPI.MPI import *
# =====================================================================================================================

def parallel_for(func, path, prefix, var=None):
	mpirank = MPI_Comm_rank()
	# print('var', var)
	if var is not None:
		i = var[mpirank]
	else:
		i = mpirank

	path_i = str(path) + '/' + str(prefix) + str(i) + '/'
	
	mkdir(path_i)
	
	os.chdir(path_i)
	# print('p=', os.getcwd(), flush=True)

	func()