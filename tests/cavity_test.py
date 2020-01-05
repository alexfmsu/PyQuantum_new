from PyQuantum.TCH.Cavity import Cavity

try:
	cv = Cavity(wc=1, wa=1, g=1, n_atoms=1)
	assert(type(cv) == type(Cavity))
except:
	print()

# cv = Cavity(wc=1, wa=1, g=1, natoms=1)

