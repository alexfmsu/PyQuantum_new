# from __future__ import division
# from __future__ import print_function
# import sys

# import numpy as np
# from mpi4py import MPI

# # from parutils import print

# comm = MPI.COMM_WORLD

# print("-"*78)
# print(" Running on %d cores" % comm.size)
# print("-"*78)

# my_N = 4
# N = my_N * comm.size

# if comm.rank == 0:
#     A = np.arange(N, dtype=np.float64)
# else:
#     A = np.empty(N, dtype=np.float64)

# my_A = np.empty(my_N, dtype=np.float64)

# # Scatter data into my_A arrays
# comm.Scatter( [A, MPI.DOUBLE], [my_A, MPI.DOUBLE] )

# print("After Scatter:")
# for r in range(comm.size):
#     if comm.rank == r:
#         print("[%d] %s" % (comm.rank, my_A))
#     comm.Barrier()

# # Everybody is multiplying by 2
# my_A *= 2

# # Allgather data into A again
# comm.Allgather( [my_A, MPI.DOUBLE], [A, MPI.DOUBLE] )

# print("After Allgather:")
# for r in range(comm.size):
#     if comm.rank == r:
#         print("[%d] %s" % (comm.rank, A))
#     comm.Barrier()

# def mpiabort_excepthook(type, value, traceback):
#     mpi_comm.Abort()
#     sys.__excepthook__(type, value, traceback)

# sys.excepthook = mpiabort_excepthook

# comm.Abort()

# sys.excepthook = sys.__excepthook__

# import sys
# import time
# import mpi4py.MPI
# mpi_comm = mpi4py.MPI.COMM_WORLD

# def mpiabort_excepthook(type, value, traceback):
#     mpi_comm.Abort()
#     sys.__excepthook__(type, value, traceback)

# def main():
#     time.sleep(5)

#     if mpi_comm.rank == 0:
#         raise ValueError('Failure')
#     print('{} continuing to execute'.format(mpi_comm.rank))
#     print('{} exiting'.format(mpi_comm.rank))

# if __name__ == "__main__":
#     # print(12)
#     sys.excepthook = mpiabort_excepthook
#     main()
#     mpi_comm.Barrier()
#     sys.excepthook = sys.__excepthook__

# # sys.excepthook = mpiabort_excepthook
# # time.sleep(5)

# # if mpi_comm.rank == 0:
# #     raise ValueError('Failure')

# # print('{} continuing to execute'.format(mpi_comm.rank))
# # print('{} exiting'.format(mpi_comm.rank))





# from PyQuantum.Tools.MPI import *

# mpirank = MPI_Comm_rank()
# mpisize = MPI_Comm_size()

# MPI_Barrier()

# if mpirank == 0:
# 	f = open('f', 'w')
# 	f.write('err'+str(mpisize))
# 	f.close()
# 	print(123)

# MPI_Barrier()
# MPI_Abort()