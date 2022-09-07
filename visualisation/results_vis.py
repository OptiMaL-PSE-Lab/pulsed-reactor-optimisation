import json
import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
import GPy 
from mpl_toolkits.axes_grid1 import make_axes_locatable



with open('outputs/initial_mf_points/dataset_x.pickle','rb') as f:
	res = pickle.load(f)
t = []
f = []
y = []
for d in res:
	t.append(d['t'])
	f.append(d['x']['fid'])
	y.append(d['N'])
t = np.cumsum(np.asarray(t))/3600

fig,axs = plt.subplots(1,1,figsize=(6,4))
axs.scatter(t,y,marker='+',c=f)
axs.set_xlabel('time (hr)')
axs.set_ylabel('Plug-flow characteristic')
plt.savefig('outputs/mf_results.png')



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
	plt.savefig('outputs/a_and_f_results.pdf')
