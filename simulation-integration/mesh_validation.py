from utils import eval_cfd_validation
import pandas as pd 
from utils import val_to_rtd 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm

exp = pd.read_csv('simulation-integration/output_validation/experiments.csv')

t1 = exp['t1'][exp['t1'].isnull()==False].values
t2 = exp['t2'][exp['t2'].isnull()==False].values
e1 = exp['e1'][exp['e1'].isnull()==False].values
e2 = exp['e2'][exp['e2'].isnull()==False].values

fig,ax = plt.subplots(1,2,figsize=(8,4))
fig.subplots_adjust(wspace=0.3)
for i in range(2):
    ax[i].set_xlabel(r'$\theta$')
    ax[i].set_ylabel(r'$E({\theta})$')
    ax[i].set_ylim(-0.1,2)

ax[0].scatter(t1,e1,c='k',s=10,marker='x',lw=0.5)
ax[1].scatter(t2,e2,c='k',s=10,marker='x',lw=0.5,label='Experiment')

fig.savefig('simulation-integration/output_validation/experimental_validation.png',dpi=800)

color = iter(cm.viridis(np.linspace(0, 1, 5)))

f2 = 0.5
for f1 in [0,0.25,0.5,0.75,1]:
    c = next(color)
    N,time,value,path = eval_cfd_validation(0.002,5,50,0.012,0.01,0.0025,0.0753,[f1,0.5])
    theta,etheta = val_to_rtd(time,value,path)
    ax[0].plot(theta,etheta,c=c,alpha=0.75)
    fig.savefig('simulation-integration/output_validation/experimental_validation.png',dpi=800)
    res1 = pd.DataFrame({'time':time,'concentration':value})
    res2 = pd.DataFrame({'theta':theta,'etheta':etheta})
    res = pd.concat([res1, res2], axis=1) 
    res.to_csv('simulation-integration/output_validation/e_1_fid_'+str(f1)+'.csv')

    N,time,value,path = eval_cfd_validation(0.004,5,50,0.012,0.01,0.0025,0.0753,[f1,0.5])
    theta,etheta = val_to_rtd(time,value,path)
    ax[1].plot(theta,etheta,c=c,alpha=0.75,label='Fidelity: '+str(f1))
    fig.savefig('simulation-integration/output_validation/experimental_validation.png',dpi=800)
    res1 = pd.DataFrame({'time':time,'concentration':value})
    res2 = pd.DataFrame({'theta':theta,'etheta':etheta})
    res = pd.concat([res1, res2], axis=1) 
    res.to_csv('simulation-integration/output_validation/e_2_fid_'+str(f1)+'.csv')
ax[1].legend()
fig.savefig('simulation-integration/output_validation/experimental_validation.png',dpi=800)