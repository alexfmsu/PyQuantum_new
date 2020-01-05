from itertools import product

n_levels = 2
n_atoms = 3
capacity = 5


def basis(capacity, n_atoms, n_levels):
	l = []

	l.append(range(capacity+1))

	for i in range(n_atoms):
		l.append(range(n_levels))

	kwargs = tuple(l)

	permutations = filter(lambda x: sum(x)<=capacity, product(*kwargs))

	return permutations

b = basis(capacity, n_atoms, n_levels)

for i in b:
	print(i)