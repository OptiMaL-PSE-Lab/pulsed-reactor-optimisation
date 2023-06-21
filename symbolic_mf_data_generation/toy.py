import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils_ed import *
import numpy as np 
from utils import * 
from main_ed import mfed,ed_hf
# from mesh_generation.coil_basic import create_mesh
import uuid
import matplotlib

x_bounds = {}
x_bounds["x1"] = [1, 10]

z_bounds = {}
z_bounds["z1"] = [0, 1]


def eval(x: dict):
    x1 = x["x1"]
    z1 = x["z1"]

    f1 = np.sin(x1) # low fidelity
    f2 = 2*np.cos(x1)/(x1) + 0.5*np.sin(3*x1)   # high fidelity

    f = (1-z1)*f1 + z1*f2

    return {"obj": f, "cost": z1**2 + x1 * 0.15, "id": str(uuid.uuid4())}

def sample_toy(xb,z):
        x_sample = {}
        x_sample['z1'] = z
        y = []
        c = []
        x = np.linspace(xb[0],xb[1], 300)
        for xi in x:
                x_sample['x1'] = xi
                e = eval(x_sample)
                y.append(e['obj'])
                c.append(e['cost'])

        return x,y,c

cmap = matplotlib.cm.get_cmap('Spectral')

            
fig,ax = plt.subplots(1,2,figsize=(10,4),sharex=True)
xb = x_bounds['x1']
x,y,c = sample_toy(xb,z_bounds['z1'][1])
ax[0].plot(x,y,c='k',lw=3,label=r'$f(x,z=1):= \frac{2\cos(x)}{x} + \frac{\sin(3x)}{2}$')
ax[1].plot(x,c,c='k',lw=3,label='$c(x,z=1)$')
x,y,c = sample_toy(xb,z_bounds['z1'][0])
ax[0].plot(x,y,c='tab:red',lw=3,label=r'$f(x,z=0):= \sin(x)$')
ax[1].plot(x,c,c='tab:red',lw=3,label='$c(x,z=0)$')
ax[1].legend(frameon=False,fontsize=8)
n = 10 
rgbr = [214, 39, 40]
rgbb = [0,0,0]
for i in np.linspace(0,1,n):
       col = np.array([i*rgbr[j] + (1-i)*rgbb[j] for j in range(3)])/256
       x,y,c = sample_toy(xb,i*z_bounds['z1'][0]+(1-i)*z_bounds['z1'][1])
       ax[0].plot(x,y,c=tuple(col),lw=3,alpha=0.2)
       ax[1].plot(x,c,c=tuple(col),lw=3,alpha=0.2)

ax[0].set_xlabel('x',fontsize=14) 
ax[0].set_ylabel('f(x)',fontsize=14) 
ax[1].set_xlabel('x',fontsize=14) 
ax[1].set_ylabel('$c(x)$',fontsize=14) 


# remmove top and right spines
for axs in ax:
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.legend(frameon=False,fontsize=8)
fig.tight_layout()
plt.savefig('symbolic_mf_data_generation/toy/vis.png',dpi=300)



hf_path = "symbolic_mf_data_generation/toy/hf_data_"
# lf_path = "symbolic_mf_data_generation/toy/lf_data.json"
mf_path = "symbolic_mf_data_generation/toy/mf_data_"

# gen_data(eval, hf_path, x_bounds, 1, 40)
# gen_data(eval, lf_path, x_bounds, 0, 100)

# path_names = {'Highest Fidelity Data': hf_path,'Lowest Fidelity Data': lf_path,'Multi-fidelity Experimental Design':mf_path}
# #path_names = {'Highest Fidelity Data': hf_path,'Lowest Fidelity Data': lf_path}
# img = 'symbolic_mf_data_generation/toy/cum_cost.png'
# plot_cum_cost(path_names,img)

for i in range(20):
        mfed(
        eval,
        mf_path+str(i)+".json",
        x_bounds,
        z_bounds,
        50,
        gamma=0.5, # weight between 0-1 of cost. 1 = only cost, 0 = only objective
        sample_initial=4,
        # sample_initial=False,
        gp_ms = 4,
        int_fidelities=[False],
        )


        ed_hf(
        eval,
        hf_path+str(i)+".json",
        x_bounds,
        z_bounds,
        50,
        sample_initial=4,
        # sample_initial=False,
        gp_ms = 4,
        int_fidelities=[False],
        )
