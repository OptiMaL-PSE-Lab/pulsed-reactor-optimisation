import numpy as np 
import GPy 
import matplotlib.pyplot as plt


def f(x,i):
	return -(x**2) - i*0.5 + np.random.uniform(0,0.5,len(x))

fig,ax = plt.subplots(2,2,figsize=(8,6))
plt.subplots_adjust(wspace=0.3,top=0.95,bottom=0.05,left=0.1,right=0.95)
for axi in ax.ravel():
	axi.set_xticks([])
cols = ['r','b','k']
gamma = [1,1.5,2]

max_aq_mf = -1e20

for i in range(3):
	y = []
	n = 5
	x = np.random.uniform(-1,1,n).reshape((n,1))
	for xi in x:
		y.append(f(xi,i))
	y = np.array(y).reshape((n,1))
	k = GPy.kern.RBF(1)+GPy.kern.White(1)
	m = GPy.models.GPRegression(x,y,k)
	m.optimize()
	m.optimize_restarts(10)
	x_plot = np.linspace(-1,1,200).reshape((200,1))
	y_mean,y_var = m.predict(x_plot)
	ax[0,0].scatter(x,y,c=cols[i])
	ax[0,0].plot(x_plot,y_mean,c=cols[i],label='Fidelity: '+str(i))
	x_plot = x_plot[:,0]
	y_mean = y_mean[:,0]
	y_var = y_var[:,0]*1.96
	ax[0,0].fill_between(x_plot,y_mean-y_var,y_mean+y_var,color=cols[i],alpha=0.1)
	ax[0,0].legend()
	ax[0,0].set_ylabel(r'$f(x)$')
	ax[0,1].set_ylabel(r'$\mu(x) + \beta^{1/2}\sigma(x)$')
	ax[1,0].set_ylabel(r'$\beta^{1/2}\sigma(x)$')
	ax[1,1].set_ylabel(r'$\gamma\cdot\beta^{1/2}\sigma(x)$')


	ax[0,1].plot(x_plot,y_mean+y_var,c=cols[i])
	ax[1,0].plot(x_plot,y_var,c=cols[i])
	ax[1,1].plot(x_plot,gamma[i]*y_var,c=cols[i])
	if i == 0:
		i_max = np.argmax(y_mean)
		ax[0,1].scatter(x_plot[i_max],y_mean[i_max]+y_var[i_max],c='r',marker='+',s=100)
		ax[1,0].scatter(x_plot[i_max],y_var[i_max],c='r',marker='+',s=100)

	if gamma[i]*y_var[i_max] > max_aq_mf:
		max_aq_mf = gamma[i]*y_var[i_max]
		i_best = i_max
		i_col = i

ax[1,1].scatter(x_plot[i_best],max_aq_mf,c=cols[i_col],marker='+')
	
plt.savefig('multi_fidelity/algorithm_demo.png')
plt.savefig('multi_fidelity/algorithm_demo.pdf')