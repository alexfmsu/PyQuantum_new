from PyQuantum.PlotBuilder.PlotBuilder3D import *
from PyQuantum.Tools.Pickle import *
import numpy as np

path = 'out/'
# path = 'out/mix/l_wc100.0_dt_l_0.001/'

t = []
l = []
g = []

lr=np.arange(0.01, 1.000, 0.01)
gr=np.arange(0.01, 0.51, 0.01)
lr = np.round(lr, 3)
gr = np.round(gr, 3)
l = list(lr)
# print(i)
# yi = pickle_load(path+'t.pkl')
# g.append(gi)	

for lk, lv in enumerate(lr):
    # print('lk:', lk)
    tl = []
    for gk, gv in enumerate(gr):
        # print('k:', lk)
        # print(path+'0.1/'+'l_'+str(lk)+'/g_'+str(gk)+'.pkl')
        # gi = pickle_load(path+'0.1/'+'l_'+str(lv)+'/g_'+str(gv)+'.pkl')
        # gi = pickle_load(path+'l0.01_0.02_0.01_g0.1_1.0_0.1_dt0.01ns_sink_limit0.95/'+'l_'+str(lk)+'/g_'+str(gk)+'.pkl')
        # print('gi:', gi)
        ti = pickle_load(path+'0.1/'+'l_'+str(lv)+'/t'+str(gv)+'.pkl')
        # ti = pickle_load(path+'l0.01_0.02_0.01_g0.1_1.0_0.1_dt0.01ns_sink_limit0.95/'+'l_'+str(lk)+'/t'+str(gk)+'.pkl')
        tl.append(ti[0])	

    t.append(tl)
    # print(np.array(t)*1e6)
        # g.append(gi)
	# for j in xi:
		# print(j)
	# print(yi)
	# x += xi
	# y += yi
# x = pickle_load(path+'g_12.pkl')
# y = pickle_load(path+'t.pkl')

# print(min(x))
# print(y)
# print(g)
g = np.array(gr)

# g*=0.01
# g = [j * 0.01 for j in i for i in g]
# t = [i * 1e6 for i in t]
t = np.array(t)
t*=1e9

# print(g)
# print(t)
# exit(0)

# data = {
# 	'x':g, 
# 	'y':l,
# 	'z':t
# }

# plt_builder = PlotBuilder3D({
# 	"title":'',
# 	"x_title":'g/hw',
# 	"y_title":'\tt, mks',
# 	"as_annotation": True,
# 	# "to_file":'1.png',
# 	"html":'1.html',
# 	"data":data
# })

# plt_builder.make_plot()
x_data=np.round(gr,3)
y_data=np.round(lr, 3)
z_data=t

print(x_data)
print(y_data)
print(z_data)

# print(g)
# print(t)
# print(lr)
# exit(0)

# exit(0)
# print(g)
        # self.x_ticktext = [str(i) for i in np.round(self.x_ticktext, 3)]

plot_builder = PlotBuilderData3D({
    'title': 'sink',

    'width': 1100,
    'height': 800,

    'to_file': False,
    'online': False,
    # 'data': data,
    'x_ticktext': [str(i) for i in np.round(x_data, 3)],
    # 'y_ticktext': [str(i) for i in np.round(y_data, 3)],
    'z_ticktext': 'l/w<sub>c</sub>',
    
    'x_title': 'g/hw',
    'y_title': 'l/hw',
    'z_title': 'time',

    'x_data': x_data,
    'y_data': y_data,
    'z_data': z_data,
    # 'y_title': 'dP/dt',
    # 'title': 'dP/dt',
    # 'title': w_0,
    'html': '3d.html',
    # 'x_range': [0, 100],
    'x_range': [min(x_data), max(x_data)],
    # 'y_range': [0, 100],
    'z_range': [0, 1],
    'y_range': [min(y_data), max(y_data)],
    # 'z_range': [min(z_data), max(z_data)],

    'y_scale': 1,

    'ticks': {
        'title': {
            'size': 20,
        },
        'family': 'Lato',
        'color': '#222',
        'size': 14,
    },
})


# plot_builder.make_plot()
plot_builder.prepare()
# plot_builder.make_plot()
