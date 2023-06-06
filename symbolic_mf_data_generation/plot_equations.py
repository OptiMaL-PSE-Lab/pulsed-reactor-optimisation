import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = 'symbolic_mf_data_generation/exp_design/mean_equation.csv'
name = 'mf_mean'
df = pd.read_csv(path)
# plot score against complexity 

com = df['Complexity'].values
loss = df['Loss'].values
eqs = df['Equation'].values

n_com = (com - min(com))/(max(com)-min(com))
n_loss = (loss - min(loss))/(max(loss)-min(loss))
dists = np.sqrt(n_com**2 + n_loss**2)
p_opt = np.argmin(dists)


fig,ax=plt.subplots(1,1,figsize=(6,4))
ax.plot(df['Complexity'],df['Loss'],lw=2,c='k')
ax.scatter(com[p_opt],loss[p_opt],s=100,c='tab:red',label='Pareto Optimal',zorder=0)
ax.text(com[p_opt]+0.1,loss[p_opt]+0.1,r'$((\sin(St)-a/0.0317)^2+1.74)^3$',fontsize=12)
ax.set_ylabel('Loss')
ax.set_xlabel('Complexity')
ax.grid(alpha=0.5)
ax.legend(frameon=False)
fig.savefig('symbolic_mf_data_generation/'+name+'_eqs.png',dpi=300,bbox_inches='tight')

