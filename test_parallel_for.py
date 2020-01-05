import os
from PyQuantum.Tools.Mkdir import *
from PyQuantum.Tools.Pickle import *
from PyQuantum.Tools.Print import print
# =====================================================================================================================
# from PyQuantum.Tools.MPI import *
from PyQuantum.Tools.MPI.MPI import *
# from PyQuantum.Tools.MPI.ParallelFor import *

# mpirank = MPI_Comm_rank()
# mpisize = MPI_Comm_size()
# =====================================================================================================================

root_path = os.getcwd()
path = 'abc'

outfile = 'slurm'
def parallel_for(var, start, end, step, func, path, prefix):
	mpirank = MPI_Comm_rank()
	
	path_i = str(path) + '/' + str(prefix) + str(mpirank) + '/'
	
	mkdir(path_i)
	
	os.chdir(path_i)
	print('cur:', os.getcwd())
	print('root:', os.path.abspath(os.curdir))
    
	func()

def foo():
	g_ = range(10)

	n1, n2 = n_batches(len(g_))

	x = 1

	try:
		for i in g_[n1:n2]:
			print(i)

			pickle_dump(i, '/' + str(i) + '.pkl')
	except Exception as e:
		os.chdir(root_path)
		MPI_Abort(err_msg=str(e), filename='s.err')

	os.chdir(root_path)
	node_print("OK", 0, filename=outfile+'.out', to_print=True)

g = range(3)


parallel_for(g, 1, 4, 1, foo, path=path, prefix='foo3_')

# =====================================================================================================================
# parallel_for(g, 1, 1, 1, foo, 'foo_')
# d = lambda: print(123)

# d()

# parallel_for(1,1,1, 
# 	lambda: ( 
# 		print(123),
# 		print(1)
# 	), g)
# =====================================================================================================================