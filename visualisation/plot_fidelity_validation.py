import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
import numpy as np 
from matplotlib import rc
import sys
import os 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import calculate_N,loss, calc_etheta
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})



fig,axs = plt.subplots(1,2,figsize=(8,3))
fig.subplots_adjust(top=0.875,bottom=0.15,left = 0.075,right=0.975)


for ax in axs:
    ax.set_ylim(0,2)
    ax.set_xlim(0.25,2.5)
    ax.set_ylabel(r'$E(\theta)$')
    ax.set_xlabel(r'$\theta$')


for e in [1,2]:
	i = e - 1
	axs[i].set_title('Experiment '+str(i+1))
	exp = pd.read_csv('data/experimental/experiments.csv')
	theta = exp['t'+str(e)]
	etheta = exp['e'+str(e)]
	theta = [t for t in theta if t >0]
	etheta = [etheta[i] for i in range(len(theta))]
	s = 10000
	n0_list = np.logspace(np.log(1), np.log(40), s)

	best = np.Inf
	for n0 in n0_list:
		l = loss(n0, theta, etheta)
		if l < best:
			best = l
			N = n0
	
	t_theta = np.copy(theta)
	t_etheta = np.zeros(len(theta))	

	for k in range(len(t_etheta)):
		t_etheta[k] = (calc_etheta(N,t_theta[k]))

	axs[i].scatter(theta,etheta,c='k',marker='o',s=40,label='Exp',lw=0,zorder=0)
	axs[i].plot(t_theta,t_etheta,c='k',label='Modelled',lw=2,ls='--')
	axs[i].legend(frameon=False)
fig.tight_layout()
plt.savefig('visualisation/n_grid.pdf',dpi=600)









fig,axs = plt.subplots(1,2,figsize=(8,3))
fig.subplots_adjust(top=0.875,bottom=0.15,left = 0.075,right=0.975)

for ax in axs:
    ax.set_ylim(0,2)
    ax.set_xlim(0.25,2)
    ax.set_ylabel(r'$E(\theta)$')
    ax.set_xlabel(r'$\theta$')

fidelities = [0,0.25,0.5,0.75,1]
color = ['r','b','y','g','r','p']
color = cm.viridis(np.linspace(0,1,len(fidelities)))
for e in [1,2]:
	i = e - 1
	axs[i].set_title('Experiment '+str(i+1))
	for j in range(len(fidelities)):
		f = fidelities[j]
		file = 'data/experimental/e_'+str(e)+'_fid_'+str(f)+'_'+str(f)+'.csv'
		df = pd.read_csv(file)
		theta = df['theta'].values
		etheta = df['etheta'].values
		axs[i].plot(theta,etheta,c=color[j],lw=2,label=r'$z_'+str(j+1)+'$')

	exp = pd.read_csv('data/experimental/experiments.csv')
	theta = exp['t'+str(e)]
	etheta = exp['e'+str(e)]
	
	axs[i].scatter(theta,etheta,c='k',marker='o',alpha=1,s=30,label='Exp',lw=0,zorder=0)
	axs[i].legend(frameon=False)
fig.tight_layout()
plt.savefig('visualisation/fidelity_validation_profile.pdf',dpi=600)


fig,axs = plt.subplots(1,2,figsize=(8 ,3))
fig.subplots_adjust(top=0.875,bottom=0.15,left = 0.075,right=0.975)

for ax in axs:
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$N$')

fidelities = [0,0.25,0.5,0.75,1]
color = ['r','b','y','g','r','p']
for e in [1,2]:
	i = e - 1
	axs[i].set_title('Experiment '+str(i+1))
	for j in range(len(fidelities)):
		f = fidelities[j]
		file = 'data/experimental/e_'+str(e)+'_fid_'+str(f)+'_'+str(f)+'.csv'
		df = pd.read_csv(file)
		theta = df['theta'].values
		etheta = df['etheta'].values
		
		N = calculate_N(df['concentration'].values,df['time'].values,path=None)
		axs[i].scatter(j,N,s=30,alpha=0.6,c='k',lw=0)

	exp = pd.read_csv('data/experimental/experiments.csv')
	theta = exp['t'+str(e)]
	etheta = exp['e'+str(e)]
	s = 10000
	n0_list = np.logspace(np.log(1), np.log(40), s)

	best = np.Inf
	for n0 in n0_list:
		l = loss(n0, theta, etheta)
		if l < best:
			best = l
			N = n0

	axs[i].plot([0,len(fidelities)-1],[N,N],c='k',lw=2,label='Exp')
	axs[i].legend(frameon=False)
	axs[i].set_xticks(np.arange(len(fidelities)),[r'$z_'+str(j+1)+'$' for j in range(len(fidelities))],label='CFD')
fig.tight_layout()
plt.savefig('visualisation/fidelity_validation_n.pdf',dpi=600)


