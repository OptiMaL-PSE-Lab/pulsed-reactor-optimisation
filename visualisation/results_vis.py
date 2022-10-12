import json
from re import I
import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
import GPy 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

import matplotlib.pyplot as plt
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})

def read_mf():
	with open('outputs/initial_mf_points/dataset_x.pickle','rb') as f:
		res = pickle.load(f)
	t = []
	f = []
	y = []

	i = 0 
	print(len(res))
	while i < len(res):
		d = res[i] 
		t.append(d['t'])
		f.append(d['x']['fid'])
		y.append(d['N'])
		i += 1
	with open('outputs/mf/logs.json','rb') as file:
		res = json.loads(file.read())

	
	for d in res.values():
		try:
			t.append(d['t'])
			f.append(d['fid'])
			y.append(d['target'])
		except KeyError:
			pass



	t = np.cumsum(np.asarray(t))/3600
	y = np.asarray(y)
	best = []
	for i in range(len(y)):
		if i == 0:
			best.append(y[i])
		else:
			if y[i] > best[i-1]:
				best.append(y[i])
			if y[i] <= best[i-1]:
				best.append(best[i-1])
	
	
	print(len(t))
	return t,y,f,best



def vis_mf():
	t,y,f,best = read_mf()
	fig,axs = plt.subplots(1,1,figsize=(6,4))
	axs.plot(t[20:],best[20:],lw=1,ls='dashed',c='k')
	axs.scatter(t,y,marker='+',c=f)
	axs.set_xlabel('time (hr)')
	axs.set_ylabel('Plug-flow characteristic')
	plt.savefig('outputs/mf_results.png')
	plt.savefig('outputs/mf_results.pdf')

#vis_mf()

def read_geom():
	with open('outputs/geom/logs.json','rb') as f:
		res = f.readlines()
	t = [] 
	N = [] 
	best = []
	for l in res:
		l = json.loads(l)
		N.append(l['target'])
		t.append(l['datetime']['delta'])
	



	t = np.cumsum(np.array(t))/3600
	# for i in range(len(t)):
	# 	if t[i] > 60:
	# 		si = i 
	# 		break 

	# N = N[:si]
	# t = t[:si]

	best = []
	for i in range(len(N)):
		if i == 0:
			best.append(N[i])
		else:
			if N[i] > best[i-1]:
				best.append(N[i])
			if N[i] <= best[i-1]:
				best.append(best[i-1])


	return t,N,best


def vis_geom():
	t,N,best = read_geom()
	fig,axs = plt.subplots(1,1,figsize=(5,3))
	fig.set_tight_layout(True)
	axs.scatter(t,N,marker='+',c='k',lw=0.75,s=40)
	axs.plot(t[8:],best[8:],lw=1,ls='--',c='k',label='Best Design')
	axs.legend(facecolor='w',framealpha=1,edgecolor='w')
	#axs.fill_betweenx([0,max(N)],min(t),t[8],color='k',alpha=0.1)
	axs.grid(True,alpha=0.3)
	axs.set_xlabel('Time (hr)')
	axs.set_ylabel('Plug-flow characteristic')
	#axs.set_ylim(0,20)
	#axs.set_xlim(0,20)
	plt.savefig('outputs/geom_results.png')
	plt.savefig('outputs/geom_results.pdf')
	return 
vis_geom()



def vis_both():
	tm,nm,f,bestm = read_mf()
	t,n,best = read_geom()
	fig,axs = plt.subplots(2,1,figsize=(7,4),sharex=True,sharey=True)
	axs[0].scatter(t,n,marker='o',s=20,color=(181/255,2/255,38/255, 1.        ),alpha=1,label='Single Fidelity')
	divider = make_axes_locatable(axs[1])
	plt.subplots_adjust(right=0.82,hspace=0.25)
	#cax = divider.append_axes('right', size='3%', pad=0.05)
	cax = plt.axes([0.85, 0.1, 0.03, 0.8])
	im = axs[1].scatter(tm,nm,marker='o',s=20,c=f,alpha=1,cmap='coolwarm',label='Multi-fidelity')


	cmap = mpl.cm.coolwarm
	bounds = np.linspace(0,1,4)
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

	fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=cax, orientation='vertical',label='Fidelity',ticks = np.linspace(0,1,5))

	axs[0].set_title('Single fidelity',fontsize=12)
	axs[1].set_title('Multi-fidelity')
	# ax_legend = fig.add_subplot()
	# ax_legend.axis('off')
	# handles_markers = []
	# markers_labels = []
	# markers_style = {'Single Fidelity':'+','Multi-fidelity':'o'}
	# for marker_name, marker_style in markers_style.items():
	# 	pts = plt.scatter([0], [0], marker=marker_style, c='black', label=marker_name)
	# 	handles_markers.append(pts)
	# 	markers_labels.append(marker_name)
	# 	pts.remove()
	# ax_legend.legend(handles_markers, markers_labels, loc='upper left', ncol=1, handlelength=1.5, handletextpad=.1,frameon=False)

	#axs[0].plot(t[8:],best[8:],lw=1,ls='dashed',c='k')
	#axs[1].plot(tm[20:],bestm[20:],lw=1,ls='dashed',c='k')
	axs[1].set_xlabel('Time (hr)')
	axs[0].set_ylabel('Plug-flow characteristic')
	axs[0].yaxis.set_label_coords(-0.075,0.0)
	plt.savefig('outputs/both.png')
	plt.savefig('outputs/both.pdf')



	fig,axs = plt.subplots(1,1,figsize=(6.5,2.5),sharex=True,sharey=True)
	divider = make_axes_locatable(axs)
	plt.subplots_adjust(right=0.88,bottom=0.2)
	axs.fill_between([0,tm[12]],0,max(nm),color='k',alpha=0.1,label='Initial Dataset')
	cax = divider.append_axes('right', size='3%', pad=0.05)
	im = axs.scatter(tm,nm,marker='o',s=20,c=f,alpha=1,cmap='coolwarm')
	cmap = mpl.cm.coolwarm
	bounds = np.linspace(0,1,4)
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
	fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=cax, orientation='vertical',label='Fidelity',ticks = np.linspace(0,1,5))
	#axs.set_title('Multi-fidelity')

	axs.plot(tm[8:],bestm[8:],lw=2,ls=':',c='k',label='Best Solution')

	#axs.text(15,11,'Initial Dataset',horizontalalignment='center',verticalalignment='center',)
	axs.set_xlabel('Time (hr)')
	axs.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=2,frameon=False)
	axs.set_ylabel('Plug-flow characteristic')
	plt.savefig('outputs/mf_results.png')
	plt.savefig('outputs/mf_results.pdf')

	return 
vis_both()

def vis_a_and_f():
	with open('outputs/a_and_f/logs.json','rb') as f:
		res = f.readlines()

	x = np.array([[1,1]])
	y = np.array([[1]])
	for l in res:
		l = json.loads(l)
		x = np.append(x,np.array([[l['params']['a'],l['params']['f']]]),axis=0)
		y = np.append(y,np.array([[l['target']]]),axis=0)
	x = x[1:,:]
	y = y[1:,:]

	ind = np.arange(len(y))
	np.random.shuffle(ind)
	x = x[ind,:]
	y = y[ind,:]

	for i in range(len(y)):
		if i == 0:
			best = [y[i]]
		else:
			if y[i] > best[i-1]:
				best.append(y[i])
			else:
				best.append(best[i-1])

	xm = np.mean(x,axis=0)
	xs = np.std(x,axis=0)
	ym = np.mean(y)
	ys = np.std(y)

	x = (x-xm)/xs
	y = (y-ym)/ys

	k = GPy.kern.RBF(2,ARD=True)
	m = GPy.models.GPRegression(x,y,k)
	m.optimize()
	m.optimize_restarts(10)

	n = 50
	a_plot = np.linspace(0.0005,0.009,n)
	f_plot = np.linspace(1,9,n)

	a_plot = (a_plot-xm[0])/xs[0]
	f_plot = (f_plot-xm[1])/xs[1]

	A,F = np.meshgrid(a_plot,f_plot)
	Z = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			Z[i,j] = m.predict(np.array([[A[i,j],F[i,j]]]))[0]

	for i in range(n):
		A[i] = (A[i]*xs[0])+xm[0]
		F[i] = (F[i]*xs[1])+xm[1]

	x = (x*xs)+xm
	y = (y*ys)+ym
	Z = (Z*ys)+ym



	fig,axs = plt.subplots(1,2,figsize=(8,3))
	plt.subplots_adjust(right=0.85,bottom=0.2)
	divider = make_axes_locatable(axs[1])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	im = axs[1].contourf(A,F,Z,50)
	fig.colorbar(im, cax=cax, orientation='vertical',label='Plug-flow performance')
	axs[1].scatter(x[:,0],x[:,1],c='w',marker='+')
	axs[1].set_xlabel('Amplitude')
	axs[1].set_ylabel('Frequency')
	axs[0].scatter(np.arange(len(y)),y,c='k',marker='+')
	axs[0].plot(np.arange(len(y)),best,c='k',ls='dashed')
	axs[0].set_xlabel('Iteration')
	axs[0].set_ylabel('Plug-flow characteristic')
	plt.savefig('outputs/a_and_f_results.png')
	plt.savefig('outputs/a_and_f_results.pdf')
