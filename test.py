# -----------------------------------------------
from PyQuantum.TCH.Cavity import Cavity
from PyQuantum.TCH.Hamiltonian import Hamiltonian
# -----------------------------------------------
from PyQuantum.TCH.WaveFunction import *
from PyQuantum.TCH.DensityMatrix import *
# -----------------------------------------------
from PyQuantum.TCH.Evolution import *
# -----------------------------------------------
from PyQuantum.TCH.Lindblad import operator_a
# -----------------------------------------------
from PyQuantum.Constants import *
# -----------------------------------------------
# cv=Cavity(wc=0.2, wa=0.2, g=[1.0, 0.5], n_atoms=2,n_levels=2)
# cv=Cavity(wc=0.2, wa=[0.2, 0.4], g=[1.0, 0.5], n_atoms=2,n_levels=2)
# cv = Cavity(wc=0.2, wa=0.2, g=1, n_atoms=2)

# cv = Cavity(wc={'0_1': 0.2, '1_2': 0.2,}, wa=[0.2, 0.2], g=[1.0, 0.5], n_atoms=2)
# import re


wa = wc
g = wa * 1e-2

cv = Cavity(
	wc={
		'0<->1': wc,
		# '0<->2': 0.2
	}, 
	wa={
		'0<->1': [wa, wa], 
		# '1<->2': [wa, wa], 
	}, 
	g={
		# '1<->2': [g, g], 
		'0<->1': [1.0, 0.5]
	}, 
	n_atoms=2, 
	n_levels=2
)

cv.info()

# exit(0)

H = Hamiltonian({
	'capacity':{
		'0<->1': 2
	}, 
	'cavity':cv, 
	'sink_dim':[1], 
	'outfile':'H.html'
})
# exit(0)

print(type(H.data))
# H = Hamiltonian(capacity={'0<->1': 2, '1<->2': 1}, cavity=cv, sink_dim=[1])

H.print_states()

H.print()

l = g * 1e-2

T = 1 * ms

# dt = 0.01 / l

dt = 10 * ns
# dt = 1 * config.ns
# dt = 1 * config.ns / 10

w0 = WaveFunction(states=H.states, init_state={'ph': [2], 'at': [0, 0], 'sink': [0]})
w0.print(header='WaveFunction:')
w0.abs_print(header='WaveFunction:')
# exit(0)


ro_0 = DensityMatrix(w0)
ro_0.print(header='DensityMatrix:')

ro_0.abs_print(header='DensityMatrix:')

a = operator_a(H)
a.print()
# exit(0)
# exit(0)

T_list = []
sink_list = []

# exit(0)

run({
    "ro_0": ro_0,
    "H": H,
    "dt": dt,
    "sink_list": sink_list,
    "T_list": T_list,
    "precision": 1e-3,
    'sink_limit': 1,
    "thres": 0.001,
    'lindblad': {
    	'out': {
            'L': operator_a(H),
            'l': l
        },
    },
})


# cv = Cavity(
# 	wc={
# 		'0_1': 0.2,
# 		'1_2': 0.2,
# 		# '0<->2': 0.2
# 	}, 
# 	wa={
# 		'0_1': [0.2, 0.2], 
# 		'1_2': [0.2, 0.2]
# 	}, 
# 	g={
# 		'0_1': [1.0, 0.5], 
# 		'1_2': [1.0, 0.5], 
# 		# '0_2': [1.0, 0.5]
# 	}, 
# 	n_atoms=2, 
# 	n_levels=3
# )



# wc_parsed = []

# for k in cv.wc.keys():
# 	groups = re.split('->|<->|-|_', k)

# 	Assert(len(groups) == 2, 'len(groups) != 2', cf())

# 	for i in range(len(groups)):
# 		groups[i] = int(groups[i])

# 		Assert(i >= 0 and i <= cv.n_levels, 'i < 0 or i > n_levels', cf())

# 	wc_parsed.append(groups)
# 	# print(k.split('_-'))
# 	print()

# for i in cv.wc_parsed:
# 	print(i)

# # exit(0)

# H = Hamiltonian(capacity={'0_1': 2}, cavity=cv)

# H.print_states()
# H = Hamiltonian(capacity={'0_1': 2, '1_2': 1}, cavity=cv)

# U = Unitary()
# cv = Cavity(wc={'0_1': 0.2, '1_2': 0.2,}, wa={'0_1': [0.2, 0.2]}, g=[1.0, 0.5], n_atoms=2)
# cv = Cavity(wc=0.2, wa=[0.2, 0.2], g=[1.0, 0.5], n_atoms=2)
# cv = Cavity(wc=1, wa=1, g=[1.0, 0.5], n_atoms=2)
# cv=Cavity(wc=1, wa=1, g=[1.0, 0.5], n_atoms=2,n_levels=3)

# cv.info()

