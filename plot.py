from PyQuantum.PlotBuilder.PlotBuilder2D import *
from PyQuantum.Tools.Pickle import *

path = 'sc/l_wc100.0_dt_l_0.001/'
# path = 'out/mix/l_wc100.0_dt_l_0.001/'

x = []
y = []

for i in range(1, 12):
	xi = pickle_load(path+'g_'+str(i)+'.pkl')
	# print(i)
	yi = pickle_load(path+'t_'+str(i)+'.pkl')
	# for j in xi:
		# print(j)
	# print(yi)
	x += xi
	y += yi
# x = pickle_load(path+'g_12.pkl')
# y = pickle_load(path+'t.pkl')

# print(min(x))
# print(y)

x = [i * 0.01 for i in x]
y = [i * 1e6 for i in y]

# print(x)
# exit(0)

data = {
	'x':x, 
	'y':y
}

plt_builder = PlotBuilder2D({
	"title":'',
	"x_title":'g/hw',
	"y_title":'\tt, mks',
	"as_annotation": True,
	# "to_file":'1.png',
	"html":'1.html',
	"data":data
})

plt_builder.make_plot()

