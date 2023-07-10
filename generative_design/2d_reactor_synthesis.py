import jax.numpy as np 
import numpy.random as rnd
import jax.random as jrnd
from jax import jit 
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax.scipy as jsp

def plot_design(d,path):
    fig,ax = plt.subplots(1,1,figsize=(1,1))
    # removing all features but the image
    ax.axis('off')
    ax.imshow(d, cmap='gray_r',interpolation='nearest',origin='lower')
    fig.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300)

@jit
def threshold(d):
    d = np.where(d > 0.5, 0, 0)
    return d

@jit
def nomalize(d):
    d = d - np.min(d)
    d = d / np.max(d)
    return d

def create_random(n):
    d = rnd.randint(0,2,(n,n))
    return d

def grf(n,r):
    d = create_random(n+r)
    x = np.linspace(-1,1,r)
    window = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
    d = jsp.signal.convolve(d, window,mode='full',method='auto')
    d = d[r:-r,r:-r]
    d = nomalize(d)
    d = threshold(d)
    return d

def add_border(d,b):
    d = np.pad(d, pad_width=b, mode='constant', constant_values=1)
    return d

def add_inlet_outlet(d,s):
    for i in range(s):
        d.at[i,0:s].set(0)
        d.at[i,-1-s:-1].set(0)
    return d



r = 8 
n = 48
b = 2
d = grf(n,r)
d = add_border(d,b)
d = add_inlet_outlet(d,2)

plot_design(d,'generative_design/2D.png')