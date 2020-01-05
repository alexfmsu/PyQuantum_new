from scipy.sparse import identity, kron, eye, csc_matrix, csr_matrix, bsr_matrix, lil_matrix
import numpy as np

m = 2
n = 2

A = lil_matrix((2, 2), dtype=str)
B = lil_matrix((2, 2), dtype=str)
# A = np.matrix([['']*n]*m, dtype=str)
# B = np.matrix([['']*n]*m, dtype=str)

for i in range(m):
	for j in range(n):
		A[i,j] = ''
		B[i,j] = ''

A[0,1] = 'acxvcxv'
B[0,1] = 'afdgsadsdsa'

for i in A.toarray():
	print(i)
print()

for i in B.toarray():
	print(i)
print()

# for i in A:
	# print(i)
# for i in B:
	# print(i)

for i in range(m):
	for j in range(n):
		# A[i,j] = A[i,j]
		A[i,j] += B[i,j]

for i in range(m):
	for j in range(n):
		print(A[i,j])
	print()

print(A[0,1])
print(A)
print('dense:')
print(A.todense())

for i in A.toarray():
	for j in i:
		print(str(j))


# m = 2
# n = 2

# a = [] * m

# for i in 
# a = lil_matrix((2, 2), dtype='O')
# b = lil_matrix((2, 2), dtype='O')
# c = lil_matrix((2, 2), dtype='O')

# a[0,1] = 'abc'
# b[1,0] = 'def'

# # print(a)
# # print(b)

# # print(a.toarray())

# # b.data=a.data+b.data

# c = a.todense()
# d = b.todense()

# c = a + b
# print(c)
# print(d)

# # for i in range(2):
# # 	for j in range(len(c)):
# # 		c[i][j] = d[i][j]

# for i in c:
# 	print(i)