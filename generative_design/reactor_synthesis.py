import jax.numpy as np 
import numpy.random as rnd
import jax.random as jrnd
from jax import jit 
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax.scipy as jsp
from skimage import measure



def plot_design(d,path):
    fig,ax = plt.subplots(1,1,figsize=(1,1))
    # removing all features but the image
    ax.axis('off')
    ax.imshow(d, cmap='gray_r',interpolation='nearest')
    fig.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300)

@jit
def threshold(d):
    d = np.where(d > 0.5, 1, 0)
    return d

@jit
def nomalize(d):
    d = d - np.min(d)
    d = d / np.max(d)
    return d

def create_random(n):
    d = rnd.randint(0,2,(n,8*n))
    return d

def grf(n,r):
    d = create_random(n+r)
    x = np.linspace(-1,1,r)
    window = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
    d = jsp.signal.convolve(d, window,mode='full',method='auto')
    d = d[r:-r,r:-r]
    d = nomalize(d)
    f_val = 0.5
    filter = np.append(np.linspace(0,f_val,int(n/2),endpoint=False),np.linspace(f_val,0,int(n/2)))
    filter_matrix = np.array([filter for i in range(len(d[0,:]))]).T
    filter_matrix = 1 - filter_matrix
    d *= filter_matrix[:-1,:]
    d = threshold(d)

    return d


def add_border(d,b):
    d = np.pad(d, pad_width=b, mode='constant', constant_values=1)
    return d

def add_inlet_outlet(d,s):
    l = len(d)
    for i in range(l-(s*2)):
        for j in range(2*s):
            d = d.at[s+i,j].set(0)
            d = d.at[s+i,-j].set(0)
    return d

def identify_holes(d):
    d_lab = measure.label(d)
    return d_lab

def remove_holes(d_lab):
    d_hist = np.bincount(d_lab.flatten())
    d_ind = np.argsort(-d_hist)[1:]
    d = np.where((d_lab == d_ind[0]) | (d_lab == d_ind[1]), 1, 0)
    return d

# r = 4
# n = 8
# b = 1
# d_store = []
# for i in tqdm(range(1000000)):
#     d = grf(n,r)
#     d = add_border(d,b)
#     d = add_inlet_outlet(d,b)
#     d = identify_holes(d)
#     d = remove_holes(d)
#     d_store.append(d)

# d_store = np.asarray(d_store)
# np.save('generative_design/2d.npy',d_store)
