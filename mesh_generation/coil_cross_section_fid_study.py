from re import A
from venv import create
import numpy as np
import sys
import os


from jax.nn import softplus
from gpjax.config import add_parameter

sys.path.insert(1, os.path.join(sys.path[0], ".."))
sys.path.insert(1, "mesh_generation/classy_blocks/src/")
from classy_blocks.classes.primitives import Edge
from classy_blocks.classes.block import Block
from classy_blocks.classes.mesh import Mesh
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import numpy as np 
import matplotlib.pyplot as plt 
import distrax as dx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import jit
from jax.config import config
from jaxtyping import Array, Float
from optax import adam
from typing import Dict
from jaxutils import Dataset
import jaxkern as jk
import gpjax as gpx
import fileinput
from PIL import Image
import imageio
from matplotlib import rc
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})



key = jr.PRNGKey(10)
# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)

def replaceAll(file,searchExp,replaceExp):
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp,replaceExp)
        sys.stdout.write(line)


def angular_distance(x, y, c):
    return jnp.abs((x - y + c) % (c * 2) - c)


class Polar(jk.base.AbstractKernel):
    def __init__(self) -> None:
        super().__init__()
        self.period: float = 2 * jnp.pi
        self.c = self.period / 2.0  # in [0, \pi]

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        tau = params["tau"]
        t = angular_distance(x, y, self.c)
        K = (1 + tau * t / self.c) * jnp.clip(1 - t / self.c, 0, jnp.inf) ** tau
        return K.squeeze()

    def init_params(self, key: jr.KeyArray) -> dict:
        return {"tau": jnp.array([4.0])}

    # This is depreciated. Can be removed once JaxKern is updated.
    def _initialise_params(self, key: jr.KeyArray) -> Dict:
        return self.init_params(key)


bij_fn = lambda x: softplus(x + jnp.array(4.0))
bij = dx.Lambda(
    forward=bij_fn, inverse=lambda y: -jnp.log(-jnp.expm1(-y - 4.0)) + y - 4.0
)

add_parameter("tau", bij)


def rotate_z(x, y, z, r_z):
    # rotation of cartesian coordinates around z axis by r_z radians
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    x_new = x * np.cos(r_z) - y * np.sin(r_z)
    y_new = x * np.sin(r_z) + y * np.cos(r_z)
    return x_new, y_new, z


def rotate_x(x0, y0, z0, r_x):
    # rotation of points around x axis by r_x radians
    y = np.array(y0) - np.mean(y0)
    z = np.array(z0) - np.mean(z0)
    y_new = y * np.cos(r_x) - z * np.sin(r_x)
    z_new = y * np.sin(r_x) - z * np.cos(r_x)
    y_new += np.mean(y0)
    z_new += np.mean(z0)
    return x0, y_new, z_new


def rotate_y(x0, y0, z0, r_y):
    # rotation of points around y axis by r_y radians
    x = np.array(x0) - np.mean(x0)
    z = np.array(z0) - np.mean(z0)
    z_new = z * np.cos(r_y) - x * np.sin(r_y)
    x_new = z * np.sin(r_y) - x * np.cos(r_y)
    x_new += np.mean(x0)
    z_new += np.mean(z0)
    return x_new, y0, z_new

def rotate_xyz(x,y,z,t,t_x,c_x,c_y,c_z):
    x -= c_x
    y -= c_y
    (
        x,
        y,
        z,
    ) = rotate_x(x, y, z, t_x)
    x, y, z = rotate_z(x, y, z, t)
    x += c_x
    y += c_y
    x, y, z = rotate_z(x, y, z, 3 * np.pi / 2)
    return x,y,z



def gp_interpolate_polar(X,y,n_interp):
    # Simulate data
    angles = jnp.linspace(0, 2 * jnp.pi, num=n_interp).reshape(-1, 1)

    D = Dataset(X=X, y=y)

    # Define polar Gaussian process 
    PKern = Polar()
    likelihood = gpx.Gaussian(num_datapoints=len(X))
    circlular_posterior = gpx.Prior(kernel=PKern) * likelihood

    # Initialise parameter state:
    parameter_state = gpx.initialise(circlular_posterior, key)
    parameter_state.params['likelihood']['obs_noise'] = 0 
    parameter_state.trainables['likelihood']['obs_noise'] = False

    # Optimise GP's marginal log-likelihood using Adam
    negative_mll = jit(circlular_posterior.marginal_log_likelihood(D, negative=True))
    optimiser = adam(learning_rate=0.05)


    inference_state = gpx.fit(
        objective=negative_mll,
        parameter_state=parameter_state,
        optax_optim=optimiser,
        num_iters=1000,
    )

    learned_params, training_history = inference_state.unpack()

    posterior_rv = likelihood(
        learned_params, circlular_posterior(learned_params, D)(angles)
    )
    mu = posterior_rv.mean()
    return angles, mu 

def create_center_circle(d,r):
    # from a centre, radius, and z rotation,
    #  create the points of a circle
    c_x, c_y, t, t_x, c_z = d
    alpha = np.linspace(0, 2 * np.pi, 63)
    z = r * np.cos(alpha) + c_z
    x = r * np.sin(alpha) + c_x
    y = [c_y for i in range(len(z))]
    x,y,z = rotate_xyz(x,y,z,t,t_x,c_x,c_y,c_z)
    return x,y,z

def create_circle(d,radius_og):
    # from a centre, radius, and z rotation,
    #  create the points of a circle
    c_x, c_y, t, t_x, c_z = d
    radius = radius_og.copy()
    angles_og = np.linspace(0,np.pi*2,len(radius),endpoint=False)
    angles = angles_og.copy()
    r_mean = np.mean(radius)
    r_std = np.std(radius)
    if r_std != 0:
        radius = (radius - r_mean)/r_std
        angles,radius = gp_interpolate_polar(angles.reshape(-1,1),radius.reshape(-1,1),63)
        radius  = radius * r_std + r_mean
    else:
        angles = np.linspace(0,np.pi*2,63,endpoint=False).reshape(-1,1)
        radius = np.array([r_mean for i in range(63)])

    for i in range(len(radius)):
        if radius[i] != radius[i]:
            radius = radius.at[i].set(r_mean)

    angles = angles[:,0]

    z_n = radius * np.cos(angles) 
    x_n = radius * np.sin(angles) 
    y_n = [c_y for i in range(len(x_n))]
    x,y,z = rotate_xyz(x_n+ c_x,y_n,z_n+c_z,t,t_x,c_x,c_y,c_z)

    # x = r × cos( θ )
    # y = r × sin( θ )
    x_p = [radius_og[i] * np.sin(angles_og[i]) + c_x for i in range(len(radius_og))] 
    z_p = [radius_og[i] * np.cos(angles_og[i]) + c_z for i in range(len(radius_og))]
    y_p = [c_y for i in range(len(radius_og))]

    x_p,y_p,z_p = rotate_xyz(x_p,y_p,z_p,t,t_x,c_x,c_y,c_z)
    return x, y, z,x_p,y_p,z_p,x_n,z_n


def cylindrical_convert(r, theta, z):
    # conversion to cylindrical coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = z
    return x, y, z

def cartesian_convert(x, y, z):
    # conversion to cartesian coordinates
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    for i in range(1,len(theta)):
        if theta[i] < theta[i-1]:
            while theta[i] < theta[i-1]:
                theta[i] = theta[i] + 2 * np.pi
    z = z
    return r, theta, z



def interpolate(y, fac_interp, kind, split_start):

    x = np.linspace(0, len(y), len(y))
    x_new = np.linspace(0, len(y), len(y) * fac_interp)
    f = interp1d(x, y, kind=kind)
    y_new = f(x_new)
    y = y_new 

    if split_start == True:
        fac = 2
        cutoff = 0.2
        x_start = np.linspace(0,int(len(y)*cutoff),int(len(y)*cutoff),endpoint=False)
        x_end = np.linspace(int(len(y)*(1-cutoff)),len(y),int(len(y)*cutoff))
        x_mid = np.linspace(int(len(y)*cutoff),int(len(y)*(1-cutoff)),len(y)-2*int(len(y)*cutoff),endpoint=False)
        x_start_new = np.linspace(x_start[0],x_start[-1],len(x_start)*2)
        f = interp1d(x_start, y[:int(len(y)*cutoff)], kind=kind)
        y_start_new = f(x_start_new)
        x_end_new = np.linspace(x_end[0],x_end[-1],len(x_end)*2)
        f = interp1d(x_end, y[len(y)-int(len(y)*cutoff):], kind=kind)
        y_end_new = f(x_end_new)
        y_new = np.concatenate((y_start_new,y[int(len(y)*cutoff):int(len(y)*(1-cutoff))],y_end_new))
        x_new = np.concatenate((x_start_new,x_mid,x_end_new))

        return y_new
    else:
        return y

def plot_block(block,ax):
    block = np.array(block)
    lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    cols = ['k','tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:gray','tab:brown']
    # ax.scatter(block[:,0],block[:,1],block[:,2],c='colks',alpha=1)
    for i in range(len(lines)):
        l = lines[i]
        ax.plot([block[l[0],0],block[l[1],0]],[block[l[0],1],block[l[1],1]],[block[l[0],2],block[l[1],2]],c='tab:blue',lw=2,alpha=0.25)
    
    return 


def create_mesh(interp_points,x_file: dict, path: str,debug: bool):
    x = x_file.copy()
    coil_rad = x["coil_rad"]
    pitch = x["pitch"]
    length = x['length']
    r_c = x['radius_center']
    s_rad = x['start_rad']

    interp_points.append(np.array([s_rad for i in range(len(interp_points[0]))])) # adding start and end inlet to be correct
    interp_points.insert(0,np.array([s_rad for i in range(len(interp_points[0]))]))


    fid_rad = int(x["fid_radial"])
    fid_ax = int(x["fid_axial"])
    fid_dt = (x["fid_deltaT"])
    fid_co = (x["fid_maxCo"])

    coils = length / (2 * np.pi * coil_rad)
    h = coils * pitch
    keys = ["x", "y", "t", "t_x", "z"]
    data = {}

    n = len(interp_points)

    t_x = -np.arctan(h / length)


    print("No inversion location specified")

    coil_vals = np.linspace(0, 2 * coils * np.pi, n)

    # x and y values around a circle
    data["x"] = [(coil_rad * np.cos(x_y)) for x_y in coil_vals] 
    data['x'] = data['x'] 
    data["y"] = [(coil_rad * np.sin(x_y)) for x_y in coil_vals]
    data['y'] = data['y']
    # rotations around z are defined by number of coils
    data["t"] = list(coil_vals)
    data["t_x"] = [t_x for i in range(n)]
    # height is linear
    data["z"] = list(np.linspace(0, h , n))

    
    mesh = Mesh()


    fig, axs = plt.subplots(1, 3, figsize=(10, 3), subplot_kw=dict(projection="3d"))
    fig.tight_layout()


    axs[0].view_init(0, 270)
    axs[1].view_init(0, 180)
    axs[2].view_init(270, 0)


    try:
        shutil.copytree("mesh_generation/mesh", path)
    except FileExistsError:
        print("Folder already exists")

    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.05, top=0.99, bottom=0.01)

    p_list = []
    p_c_list = []
    p_interp = []

    for i in tqdm(range(n)):
        x, y, z,x_p,y_p,z_p,x_n,z_n = create_circle(
            [data[keys[j]][i] for j in range(len(keys))],interp_points[i]
        )
        for ax in axs:
            ax.scatter(x_p, y_p, z_p, c="k", alpha=1,s=10)
            
        x_c, y_c, z_c = create_center_circle(
            [data[keys[j]][i] for j in range(len(keys))],r_c
        )
        p_list.append([x,y,z])
        p_interp.append([x_n,z_n])
        p_c_list.append([x_c,y_c,z_c])

    for ax in axs:
        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid()
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    axs[0].set_xlabel("x",fontsize=14)
    axs[0].set_zlabel("z",fontsize=14)

    axs[1].set_ylabel("y",fontsize=14)
    axs[1].set_zlabel("z",fontsize=14)

    axs[2].set_ylabel("y",fontsize=14)
    axs[2].set_xlabel("x",fontsize=14)
    plt.savefig(path+'/points.pdf', dpi=600)
    p_list = np.asarray(p_list)
    p_c_list = np.asarray(p_c_list)
    p_interp = np.asarray(p_interp)

    for i in range(n):
        for ax in axs:
            ax.plot(p_list[i,0,:],p_list[i,1,:],p_list[i,2,:], c="k", alpha=1,lw=1)
    
    for ax in axs:
        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid()
    axs[0].set_xlabel("x",fontsize=14)
    axs[0].set_zlabel("z",fontsize=14)

    axs[1].set_ylabel("y",fontsize=14)
    axs[1].set_zlabel("z",fontsize=14)

    axs[2].set_ylabel("y",fontsize=14)
    axs[2].set_xlabel("x",fontsize=14)
    fig.savefig(path+'/gp_slices.pdf', dpi=600)

    figc,axsc = plt.subplots(2,int(len(interp_points)/2),figsize=(10,6),subplot_kw=dict(projection='polar'),sharey=True)
    figc.tight_layout()
    gridspec = fig.add_gridspec(1, 1)
    angles = np.linspace(0,2*np.pi,len(interp_points[0]),endpoint=False)

    i = 0 
    for ax in axsc.ravel():
        ax.set_yticks([0,0.001,0.002,0.003,0.004],[0,'1E-3','2E-3','3E-3','4E-3'],fontsize=8)
        ax.set_xticks(np.linspace(0,2*np.pi,8,endpoint=False),['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$'])

        ax.set_ylim(0,0.004)
        ax.scatter(angles, interp_points[i], alpha=1,c='k')
        x = p_interp[i,0,:] 
        z = p_interp[i,1,:] 
        #convert x and z to polar coordinates
        r = np.sqrt(x**2 + z**2)
        theta = np.arctan2(x,z)

        ax.plot(theta, r, alpha=1,c='k',lw=2.5)
        i += 1

    figc.savefig(path+'/points_short.pdf', dpi=600)




    p_cylindrical_list = []
    p_c_cylindrical_list = []

    for i in range(len(p_list[0,0,:])):
        r,theta,z = cartesian_convert(p_list[:,0,i],p_list[:,1,i],p_list[:,2,i])
        r_c,theta_c,z_c = cartesian_convert(p_c_list[:,0,i],p_c_list[:,1,i],p_c_list[:,2,i])
        p_cylindrical_list.append([r,theta,z])
        p_c_cylindrical_list.append([r_c,theta_c,z_c])

    p_cylindrical_list = np.asarray(p_cylindrical_list)
    p_c_cylindrical_list = np.asarray(p_c_cylindrical_list)
    p_new_list = [] 
    p_c_new_list = []
    for i in range(len(p_cylindrical_list[:,0,0])):

        r = interpolate(p_cylindrical_list[i,0,:], fid_ax, "quadratic",split_start=True)
        theta = interpolate(p_cylindrical_list[i,1,:], fid_ax, "quadratic",split_start=True)
        z = interpolate(p_cylindrical_list[i,2,:], fid_ax, "quadratic",split_start=True)
        p_new_list.append([r,theta,z])
        r_c = interpolate(p_c_cylindrical_list[i,0,:], fid_ax, "quadratic",split_start=True)
        theta_c = interpolate(p_c_cylindrical_list[i,1,:], fid_ax, "quadratic",split_start=True)
        z_c = interpolate(p_c_cylindrical_list[i,2,:], fid_ax, "quadratic",split_start=True)
        p_c_new_list.append([r_c,theta_c,z_c])

    p_new_list = np.asarray(p_new_list)
    p_c_new_list = np.asarray(p_c_new_list)

    p_list = []
    p_c_list = []
    for i in range(len(p_new_list[:,0,0])):
        x,y,z = cylindrical_convert(p_new_list[i,0,:],p_new_list[i,1,:],p_new_list[i,2,:])

        x_c,y_c,z_c = cylindrical_convert(p_c_new_list[i,0,:],p_c_new_list[i,1,:],p_c_new_list[i,2,:])
        p_list.append([x,y,z])
        p_c_list.append([x_c,y_c,z_c])

    p_list = np.asarray(p_list)
    p_c_list = np.asarray(p_c_list)

    for i in range(len(p_list[:,0,0])):
        for ax in axs:
            ax.plot(p_list[i,0,:],p_list[i,1,:],p_list[i,2,:], c="k", alpha=0.5,lw=0.5)

    for ax in axs:
        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid()
    axs[0].set_xlabel("x",fontsize=14)
    axs[0].set_zlabel("z",fontsize=14)

    axs[1].set_ylabel("y",fontsize=14)
    axs[1].set_zlabel("z",fontsize=14)

    axs[2].set_ylabel("y",fontsize=14)
    axs[2].set_xlabel("x",fontsize=14)
    fig.savefig(path+'/interpolated.pdf', dpi=600)

    # [circle points, coordinates, number of circles]

    fig_i, axs_i = plt.subplots(1, 3, figsize=(10, 3), subplot_kw=dict(projection="3d"))
    fig_i.tight_layout()

    axs_i[0].view_init(0, 270)
    axs_i[1].view_init(0, 180)
    axs_i[2].view_init(270, 0)

    for i in range(len(p_list[:,0,i])):
        for ax in axs_i:
            ax.plot(p_list[i,0,:],p_list[i,1,:],p_list[i,2,:], c="k", alpha=0.5,lw=0.5)

    for i in range(len(p_list[i,0,:])):
        for ax in axs_i:
            ax.plot(p_list[:,0,i],p_list[:,1,i],p_list[:,2,i], c="k", alpha=0.5,lw=0.5)

    for ax in axs_i:
        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid()
    axs_i[0].set_xlabel("x",fontsize=14)
    axs_i[0].set_zlabel("z",fontsize=14)
    axs_i[1].set_ylabel("y",fontsize=14)
    axs_i[1].set_zlabel("z",fontsize=14)
    axs_i[2].set_ylabel("y",fontsize=14)
    axs_i[2].set_xlabel("x",fontsize=14)
    plt.savefig(path+'/interpolated_clean.pdf', dpi=600)

    if debug == True:
        p_list= p_list[:,:,:3]
        p_c_list= p_c_list[:,:,:3]


    col = (212/255,41/255,144/255)
    col = 'k'

    fig, axs = plt.subplots(1, 3, figsize=(10, 3), subplot_kw=dict(projection="3d"))
    fig.tight_layout()

    axs[0].view_init(0, 270)
    axs[1].view_init(0, 180)
    axs[2].view_init(270, 0)

    for ax in axs:
        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    axs[0].set_xlabel("x",fontsize=14)
    axs[0].set_zlabel("z",fontsize=14)
    axs[1].set_ylabel("y",fontsize=14)
    axs[1].set_zlabel("z",fontsize=14)
    axs[2].set_ylabel("y",fontsize=14)
    axs[2].set_xlabel("x",fontsize=14)

    for i in range(len(p_list[0,0,:])):
        if i < 3:
            for ax in axs:
                ax.plot(p_list[:,0,i],p_list[:,1,i],p_list[:,2,i], c=col, alpha=1,lw=2)

        spacing = (len(p_list[:,0,i])+1)/8
        starts = np.append(np.linspace(0,(len(p_list[:,0,i])+1),4,endpoint=False),0)

        for seg in range(4): 
            s_indices = [starts[seg],starts[seg]+spacing,starts[seg+1]]

            if i != len(p_list[0,0,:])-1:

                p1 = [list(p_c_list[int(s),:,i]) for s in s_indices]
                p2 = [list(p_c_list[int(s),:,i+1]) for s in s_indices]
                c1 = list(np.mean(p_c_list[:,:,i],axis=0))
                c2 = list(np.mean(p_c_list[:,:,i+1],axis=0))
                p1.append(c1)
                p2.append(c2)
                block_points = p2+p1
                if i < 2:
                    for ax in axs:
                        plot_block(block_points,ax)

                # for ax in axs:
                #     plot_block(block_points,ax)

                block_edges = [
                        Edge(4, 5, None), 
                        Edge(5, 6, None),
                        Edge(0, 1, None),
                        Edge(1, 2, None),
                    ]
                block = Block.create_from_points(block_points, block_edges)
                block.chop(0,count=fid_rad)
                block.chop(1,count=fid_rad)
                block.chop(2,count=1)
                if i == 0:
                    block.set_patch("top", "inlet")
                if i == len(p_list[0,0,:])-2:
                    block.set_patch("bottom", "outlet")
                mesh.add_block(block)

                p1_inner = [list(p_c_list[int(s),:,i]) for s in [s_indices[0],s_indices[1]]]
                p1_outer = [list(p_list[int(s),:,i]) for s in [s_indices[1],s_indices[0]]]
                p2_inner = [list(p_c_list[int(s),:,i+1]) for s in [s_indices[0],s_indices[1]]]
                p2_outer = [list(p_list[int(s),:,i+1]) for s in [s_indices[1],s_indices[0]]]
                block_points = p1_inner+p1_outer+p2_inner+p2_outer

                if i < 2:
                    for ax in axs:
                        plot_block(block_points,ax)
                # for ax in axs:
                #     plot_block(block_points,ax)

                spline_indices = np.arange(starts[seg],starts[seg]+spacing)
                spline_1 = [list(p_list[int(s),:,i+1]) for s in spline_indices]
                spline_2 = [list(p_list[int(s),:,i]) for s in spline_indices]
                spline_1.reverse()
                spline_2.reverse()

                block_edges = [
                        Edge(6, 7, spline_1),
                        Edge(2, 3, spline_2)
                    ]
                
                block = Block.create_from_points(block_points, block_edges)
                block.chop(0,count=fid_rad)
                block.chop(1,count=fid_rad)
                block.chop(2,count=1)
                if i == 0:
                    block.set_patch("bottom", "inlet")
                if i == len(p_list[0,0,:])-2:
                    block.set_patch("top", "outlet")
                block.set_patch(["back"], "wall")
                mesh.add_block(block)


                p1_inner = [list(p_c_list[int(s),:,i]) for s in [s_indices[1],s_indices[2]]]
                p1_outer = [list(p_list[int(s),:,i]) for s in [s_indices[2],s_indices[1]]]
                p2_inner = [list(p_c_list[int(s),:,i+1]) for s in [s_indices[1],s_indices[2]]]
                p2_outer = [list(p_list[int(s),:,i+1]) for s in [s_indices[2],s_indices[1]]]
                block_points = p1_inner+p1_outer+p2_inner+p2_outer

                if i < 2:
                    for ax in axs:
                        plot_block(block_points,ax)
                # for ax in axs:
                #     plot_block(block_points,ax)

                if seg == 3:
                    spline_indices = np.append(np.arange(starts[seg]+spacing,starts[seg]+spacing+spacing-1),0)
                else:
                    spline_indices = np.arange(starts[seg]+spacing,starts[seg+1])
                spline_1 = [list(p_list[int(s),:,i+1]) for s in spline_indices]
                spline_2 = [list(p_list[int(s),:,i]) for s in spline_indices]
                spline_1.reverse()
                spline_2.reverse()
                block_edges = [
                        Edge(6, 7, spline_1),
                        Edge(2, 3, spline_2),
                    ]
                
                block = Block.create_from_points(block_points, block_edges)
                block.chop(0,count=fid_rad)
                block.chop(1,count=fid_rad)
                block.chop(2,count=1)
                if i == 0:
                    block.set_patch("bottom", "inlet")
                if i == len(p_list[0,0,:])-2:
                    block.set_patch("top", "outlet")
                block.set_patch(["back"], "wall")
                mesh.add_block(block)
    for ax in axs:
        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    axs[0].set_xlabel("x",fontsize=14)
    axs[0].set_zlabel("z",fontsize=14)
    axs[1].set_ylabel("y",fontsize=14)
    axs[1].set_zlabel("z",fontsize=14)
    axs[2].set_ylabel("y",fontsize=14)
    axs[2].set_xlabel("x",fontsize=14)

    plt.savefig(path+'/blocks.pdf', dpi=600)


    # adding deltaT to controlDict
    replaceAll(path+"/system/controlDict","deltaT          0.000001;","deltaT          "+str(fid_dt)+";")
    replaceAll(path+"/system/controlDict","maxCo           5.0;","maxCo          "+str(fid_co)+";")


    print("Writing geometry")
    mesh.write(output_path=os.path.join(path, "system", "blockMeshDict"), geometry=None)
    print("Running blockMesh")
    os.system("chmod +x " + path + "/Allrun.mesh")
    os.system(path + "/Allrun.mesh")
    return

# n = 10
# n_cross_section = 8
# coil_data = {"length":0.0753,"pitch":0.0075,"coil_rad":0.0085,"fid_axial":5,"fid_radial":2}

# cross_section_points = [np.random.uniform(0.001,0.004,n_cross_section) for i in range(n)]
# cross_section = [np.array([0.0025 for i in range(n_cross_section)]) for i in range(n)]
# a = np.linspace(0,1,30)
# for i in range(30):
#     combo = [(1-a[i])*cross_section[j][:] + a[i]*cross_section_points[j][:] for j in range(n)]
#     create_mesh(combo,coil_data,'mesh_generation/test/'+str(i)+'.pdf')



# images = [] # creating image array
# for i in range(30): # iterating over images
#     im = Image.open('mesh_generation/test/'+str(i)+'.pdf')
#     images.append(im)
#     os.remove('mesh_generation/test/'+str(i)+'.pdf') # this then deletes the image file from the folder
# imageio.mimsave('mesh_generation/test/res.gif', images, format='GIF-PIL', quantizer=0) # this then saves the array of images as a gif


# -----------------------------

# n_circ = 4
# n_cross_section = 6
# coils = 2 
# length = np.pi * 2 * 0.0125 * coils
# key = jr.PRNGKey(10)
# coil_data = {"start_rad":0.0025,"radius_center":0.0015,"length":length,"a": 0.0009999999310821295, "f": 2.0, "re": 50.0, "pitch": 0.010391080752015114, "coil_rad": 0.012500000186264515, "inversion_loc": 0.6596429944038391, "fid_axial": 10, "fid_radial": 3}
# cross_section_points = [np.random.uniform(0.002,0.004,n_cross_section) for i in range(n_circ)]
# # cross_section_points = [np.array([0.0025 for i in range(n_cross_section)]) for i in range(n)]
# create_mesh(cross_section_points,coil_data,'mesh_generation/test/',debug=False)

# n_circ = 6
# n_cross_section = 6
# coils = 3
# length = np.pi * 2 * 0.0125 * coils
# coil_data = {"start_rad":0.0025,"radius_center":0.00125,"length":length,"a": 0.0009999999310821295, "f": 2.0, "re": 50.0, "pitch": 0.010391080752015114, "coil_rad": 0.012500000186264515, "inversion_loc": 0.6596429944038391, "fid_axial": 50, "fid_radial": 5}
# x = {"r_0_0": 0.004, "r_0_1": 0.002, "r_0_2": 0.004, "r_0_3": 0.004, "r_0_4": 0.002, "r_0_5": 0.004, "r_1_0": 0.004, "r_1_1": 0.002, "r_1_2": 0.002, "r_1_3": 0.002, "r_1_4": 0.004, "r_1_5": 0.004, "r_2_0": 0.002, "r_2_1": 0.0031486782341314137, "r_2_2": 0.002, "r_2_3": 0.004, "r_2_4": 0.0023774798948272657, "r_2_5": 0.004, "r_3_0": 0.002, "r_3_1": 0.004, "r_3_2": 0.002, "r_3_3": 0.002, "r_3_4": 0.002, "r_3_5": 0.004, "r_4_0": 0.002, "r_4_1": 0.004, "r_4_2": 0.004, "r_4_3": 0.004, "r_4_4": 0.004, "r_4_5": 0.004, "r_5_0": 0.002, "r_5_1": 0.004, "r_5_2": 0.004, "r_5_3": 0.002392031371596949, "r_5_4": 0.004, "r_5_5": 0.002, "fid_axial": 10.0, "fid_radial": 2.0}
# coil_data['fid_radial'] = x['fid_radial']
# coil_data['fid_axial'] = x['fid_axial']

# x_list = []
# for i in range(n_circ):
#     x_add = []
#     for j in range(n_cross_section):
#         x_add.append(x['r_' + str(i) + '_' + str(j)])

#     x_list.append(np.array(x_add))

# a = 0
# f = 0
# re = 50
# case = 'mesh_generation/test'

# create_mesh(x_list,coil_data.copy(),case,debug=False)