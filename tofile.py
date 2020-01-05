from PyQuantum.Tools.Matrix import *

A = Matrix(m=2, n=2, dtype=np.complex64, data=lil_matrix((2,2)))
A.data[0,0] = 1

A.print()

A.tofile_abs('A', sep=' ')