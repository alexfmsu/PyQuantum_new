from PyQuantum.Tools.Pickle import *

data = pickle_load('l_wc/ampl01.pkl')

info = pickle_load('l_wc/info.pkl')

for i in data:
	print(i)

print()

for k, v in info.items():
	print(k, ': ', v, sep='')

