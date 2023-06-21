import sys
import os
from utils_ed import * 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from symbolic_mf_data_generation.main_ed import mfed
from mesh_generation.coil_basic import create_mesh
from jax import grad, jit, value_and_grad
import jax.numpy as jnp
from utils_plotting import *
from utils_gp import *

paths = ['symbolic_mf_data_generation/toy/mf_data.json']
for p in paths:
    data = read_json(p)['data']

    mse = []
    for d in data:
        try:
            mse.append(d['MSE'])
        except:
            pass
        
fig,ax = plt.subplots(1,1,figsize=(8,4))
ax.plot(np.arange(len(mse)),mse,c='k',lw=3)
ax.set_xlabel('MSE')
ax.set_ylabel('Iteration')
# remove upper and right axis 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()
fig.savefig('symbolic_mf_data_generation/toy/mse.png',dpi=300)
