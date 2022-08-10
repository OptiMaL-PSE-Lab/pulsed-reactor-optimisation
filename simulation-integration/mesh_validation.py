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

fig,ax = plt.subplots(1,2,figsize=(6,3))
fig.subplots_adjust(wspace=0.3,bottom=0.2,right=0.95)
for i in range(2):
    ax[i].set_xlabel(r'$\theta$')
    ax[i].set_ylabel(r'$E({\theta})$')
    ax[i].set_ylim(-0.1,2.5)

ax[0].scatter(t1,e1,c='k',s=10,marker='x',lw=0.5)
ax[1].scatter(t2,e2,c='k',s=10,marker='x',lw=0.5,label='Experiment')


color = iter(cm.viridis_r(np.linspace(0, 1, 3)))


for f in [0,0.5,1]:

    c = next(color)
    # N,time,value,path = eval_cfd_validation(0.002,5,50,0.012,0.01,0.0025,0.0753,[f,f])
    # theta,etheta = val_to_rtd(time,value,path)
    df = pd.read_csv('simulation-integration/output_validation/e_1_fid_'+str(f)+'.csv')
    theta = df['theta'].values
    etheta = df['etheta'].values

    ax[0].plot(theta,etheta,c=c,alpha=0.75)

    # res1 = pd.DataFrame({'time':time,'concentration':value})
    # res2 = pd.DataFrame({'theta':theta,'etheta':etheta})
    # res = pd.concat([res1, res2], axis=1) 
    # res.to_csv('simulation-integration/output_validation/e_1_fid_'+str(f)+'.csv')

    # N,time,value,path = eval_cfd_validation(0.004,5,50,0.012,0.01,0.0025,0.0753,[f,f])
    # theta,etheta = val_to_rtd(time,value,path)
    df = pd.read_csv('simulation-integration/output_validation/e_2_fid_'+str(f)+'.csv')
    theta = df['theta'].values
    etheta = df['etheta'].values
    ax[1].plot(theta,etheta,c=c,alpha=0.75,label='Fidelity: '+str(f))
    ax[0].grid(alpha=0.5)
    ax[1].grid(alpha=0.5)
    # res1 = pd.DataFrame({'time':time,'concentration':value})
    # res2 = pd.DataFrame({'theta':theta,'etheta':etheta})
    # res = pd.concat([res1, res2], axis=1) 
    # res.to_csv('simulation-integration/output_validation/e_2_fid_'+str(f)+'.csv')

ax[1].legend()
fig.savefig('simulation-integration/output_validation/experimental_validation.png',dpi=800)