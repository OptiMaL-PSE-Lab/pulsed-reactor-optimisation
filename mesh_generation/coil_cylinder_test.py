from re import A
from venv import create
import numpy as np
import sys
import os
import math 
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

from PIL import Image
import imageio
from matplotlib import rc
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})



key = jr.PRNGKey(10)
# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


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


def unit_vector(vector):
    # returns a unit vector
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    # angle between two vectors
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


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

def create_center_circle(d1, d2,r_c):
    # takes 2 cylindrical coordinates
    # and rotates the location of the second to be orthogonal
    # to the vector between the centre of the two circles

    # circle_test.py provides an example
    r1, t1, z1, rad1 = d1
    r2, t2, z2, rad2 = d2
    c_x1, c_y1, c_z1 = cylindrical_convert(r_c, t1, z1)
    c_x2, c_y2, c_z2 = cylindrical_convert(r_c, t2, z2)
    alpha = np.linspace(0, 2 * np.pi, 63)
    y1 = rad1 * np.cos(alpha) + c_y1
    x1 = rad1 * np.sin(alpha) + c_x1
    z1 = [c_z1 for i in range(len(x1))]
    y2 = rad2 * np.cos(alpha) + c_y2
    x2 = rad2 * np.sin(alpha) + c_x2
    z2 = [c_z2 for i in range(len(x2))]
    c1 = np.mean([x1, y1, z1], axis=1)
    c2 = np.mean([x2, y2, z2], axis=1)
    x1, y1, z1 = np.array([x1, y1, z1]) - np.array([c1 for i in range(len(x1))]).T
    x2, y2, z2 = np.array([x2, y2, z2]) - np.array([c1 for i in range(len(x1))]).T
    v = c2 - c1
    a_z = angle_between([0, 0, 1], v)
    xv = [v[0], v[1]]
    b = [0, 1]
    a_x = math.atan2(xv[1] * b[0] - xv[0] * b[1], xv[0] * b[0] + xv[1] * b[1])
    x2p, y2p, z2p = rotate_x(x2, y2, z2, (-a_z))
    x2p, y2p, z2p = rotate_z(x2p, y2p, z2p, a_x)
    return x2p + c1[0], y2p + c1[1], z2p + c1[2]


def create_circle(d1, d2):
    # takes 2 cylindrical coordinates
    # and rotates the location of the second to be orthogonal
    # to the vector between the centre of the two circles

    # circle_test.py provides an example
    r1, t1, z1, rad1 = d1
    r2, t2, z2, rad2 = d2
    c_x1, c_y1, c_z1 = cylindrical_convert(r1, t1, z1)
    c_x2, c_y2, c_z2 = cylindrical_convert(r2, t2, z2)
    alpha = np.linspace(0, 2 * np.pi, 63)
    y1 = rad1 * np.cos(alpha) + c_y1
    x1 = rad1 * np.sin(alpha) + c_x1
    z1 = [c_z1 for i in range(len(x1))]
    y2 = rad2 * np.cos(alpha) + c_y2
    x2 = rad2 * np.sin(alpha) + c_x2
    z2 = [c_z2 for i in range(len(x2))]
    c1 = np.mean([x1, y1, z1], axis=1)
    c2 = np.mean([x2, y2, z2], axis=1)
    x1, y1, z1 = np.array([x1, y1, z1]) - np.array([c1 for i in range(len(x1))]).T
    x2, y2, z2 = np.array([x2, y2, z2]) - np.array([c1 for i in range(len(x1))]).T
    v = c2 - c1
    a_z = angle_between([0, 0, 1], v)
    xv = [v[0], v[1]]
    b = [0, 1]
    a_x = math.atan2(xv[1] * b[0] - xv[0] * b[1], xv[0] * b[0] + xv[1] * b[1])
    x2p, y2p, z2p = rotate_x(x2, y2, z2, (-a_z))
    x2p, y2p, z2p = rotate_z(x2p, y2p, z2p, a_x)
    return x2p + c1[0], y2p + c1[1], z2p + c1[2]

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



def interpolate(y, fac_interp, kind):

    x = np.linspace(0, len(y), len(y))
    x_new = np.linspace(0, len(y), len(y) * fac_interp)
    f = interp1d(x, y, kind=kind)
    y_new = f(x_new)
    y = y_new 

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


def add_start_end(x,y,z,t,t_x,L):
    x = np.array(x) 
    y = np.array(y)
    z = np.array(z)
    rho,theta,z = cartesian_convert(x,y,z)
    rho_start = np.sqrt(rho[0]**2 + L**2)
    theta_start = theta[0] - np.arctan(L/rho[0])
    z_start = z[0]


    rho_end = np.sqrt(rho[-1]**2 + L**2)
    theta_end = theta[-1] + np.arctan(L/rho[-1])
    z_end = z[-1]
    rho = np.append(np.append(rho_start,rho),rho_end)
    theta = np.append(np.append(theta_start,theta),theta_end)
    z = np.append(np.append(z_start,z),z_end)
    t = np.append(np.append(t[0],t),t[-1])
    t_x = np.append(np.append(t_x[0],t_x),t_x[-1])
    x,y,z = cylindrical_convert(rho,theta,z)

    return x,y,z,t,t_x

def interpolate_split(r,theta,z,fid_ax):
    #z[1] = (z[0]+z[2])/2
    x,y,z = cylindrical_convert(r,theta,z)
    r = interpolate(r[1:-1], fid_ax, "quadratic")
    theta = interpolate(theta[1:-1], fid_ax, "quadratic")
    z_c = interpolate(z[1:-1], fid_ax, "quadratic")

    x_start = interpolate([x[0],x[1]], int(fid_ax/2), "linear")
    y_start = interpolate([y[0],y[1]], int(fid_ax/2), "linear")
    z_start = interpolate([z[0],z[1]], int(fid_ax/2), "linear")
    r_start,theta_start,z_start = cartesian_convert(x_start,y_start,z_start)

    x_end = interpolate([x[-1],x[-2]], int(fid_ax/2), "linear")
    y_end = interpolate([y[-1],y[-2]], int(fid_ax/2), "linear")
    z_end = interpolate([z[-1],z[-2]], int(fid_ax/2), "linear")
    r_end,theta_end,z_end = cartesian_convert(x_end,y_end,z_end)

    r = np.append(np.append(r_start[:-1],r),np.flip(r_end[:-1]))
    theta = np.append(np.append(theta_start[:-1],theta),np.flip(theta_end[:-1]))
    z_c = np.append(np.append(z_start[:-1],z_c),np.flip(z_end[:-1]))

    # r = np.append(r_start,r)
    # theta = np.append(theta_start,theta)
    # z_c = np.append(z_start,z_c)

    return r,theta,z_c

def create_mesh(data,path,n,nominal_data):



    var_keys = ["rho", "z"]
    extra_keys = ['theta', 'tube_rad']
    keys = var_keys + extra_keys

    rad = nominal_data['tube_rad_0']
    x = {}
    for k in var_keys:
        add = np.array([data[k+'_'+str(i)] for i in range(n)])
        nominal = np.array([nominal_data[k+'_'+str(i)] for i in range(n)])
        x[k] = nominal + add   
    for k in extra_keys:
        x[k] = np.array([nominal_data[k+'_'+str(i)] for i in range(n)])

    interp_points = [np.array([rad for i in range(8)]) for i in range(n+2)]

    fid_rad = int(data["fid_radial"])
    fid_ax = int(data["fid_axial"])


    print("No inversion location specified")

    # x and y values around a circle
    
    # data["x"] = [(coil_rad * np.cos(x_y)) for x_y in coil_vals] 
    # data['x'] = data['x'] 
    # data["y"] = [(coil_rad * np.sin(x_y)) for x_y in coil_vals]
    # data['y'] = data['y']
    # # rotations around z are defined by number of coils
    # data["t"] = list(coil_vals)
    # data["t_x"] = [0]+[t_x for i in range(n-2)]+[0]
    # # height is linear
    # data["z"] = list(np.linspace(0, h , n))
    data = {}

    n = len(interp_points)-2
    data["t"] = x['theta']
    x,y,z = cylindrical_convert(x['rho'],x['theta'],x['z'])
    data['x'] = x 
    data['y'] = y
    data['z'] = z
    data['t_x'] = [0]+[0 for i in range(len(x)-2)]+[0]

    L = nominal_data['rho_0']
    data['x'],data['y'],data['z'],data['t'],data['t_x'] = add_start_end(data['x'],data['y'],data['z'],data['t'],data['t_x'],L)

    keys = ["x", "y", "t", "t_x", "z"]
    
    mesh = Mesh()

    try:
        shutil.copytree("mesh_generation/mesh", path)
    except FileExistsError:
        print("Folder already exists")

    p_list = []
    p_c_list = []
    r_c = nominal_data['r_c']
    data['rho'],data['theta'],data['z'] = cartesian_convert(data['x'],data['y'],data['z'])

    data['rho'],data['theta'],data['z'] = interpolate_split(data['rho'],data['theta'],data['z'],fid_ax)

    n = len(data['rho'])
    d_keys = ['rho', 'theta','z']
    for i in tqdm(range(1,n)):

        x, y, z = create_circle(
            [data[d_keys[j]][i-1] for j in range(len(d_keys))]+[rad],
            [data[d_keys[j]][i] for j in range(len(d_keys))]+[rad]
        )

        x_c, y_c, z_c = create_center_circle(
            [data[d_keys[j]][i-1] for j in range(len(d_keys))]+[rad],[data[d_keys[j]][i] for j in range(len(d_keys))]+[rad],r_c
        )

        p_list.append([x,y,z])
        p_c_list.append([x_c,y_c,z_c])


    p_list = np.asarray(p_list)
    p_c_list = np.asarray(p_c_list)

    p_cylindrical_list = []
    p_c_cylindrical_list = []

    for i in range(len(p_list[0,0,:])):
        r,theta,z = cartesian_convert(p_list[:,0,i],p_list[:,1,i],p_list[:,2,i])
        r_c,theta_c,z_c = cartesian_convert(p_c_list[:,0,i],p_c_list[:,1,i],p_c_list[:,2,i])
        p_cylindrical_list.append([r,theta,z])
        p_c_cylindrical_list.append([r_c,theta_c,z_c])

    # p_cylindrical_list = np.asarray(p_cylindrical_list)
    # p_c_cylindrical_list = np.asarray(p_c_cylindrical_list)
    # p_new_list = [] 
    # p_c_new_list = []
    # for i in range(len(p_cylindrical_list[:,0,0])):

    #     r,theta,z = interpolate_split(p_cylindrical_list[i,0,:],p_cylindrical_list[i,1,:],p_cylindrical_list[i,2,:], fid_ax)
    #     p_new_list.append([r,theta,z])
    #     r_c,theta_c,z_c = interpolate_split(p_c_cylindrical_list[i,0,:],p_c_cylindrical_list[i,1,:],p_c_cylindrical_list[i,2,:], fid_ax)
    #     p_c_new_list.append([r_c,theta_c,z_c])


    # p_new_list = np.asarray(p_new_list)
    # p_c_new_list = np.asarray(p_c_new_list)
    p_new_list = np.array(p_cylindrical_list.copy())
    p_c_new_list = np.array(p_c_cylindrical_list.copy())
    p_list = []
    p_c_list = []
    for i in range(len(p_new_list[:,0,0])):
        x,y,z = cylindrical_convert(p_new_list[i,0,:],p_new_list[i,1,:],p_new_list[i,2,:])

        x_c,y_c,z_c = cylindrical_convert(p_c_new_list[i,0,:],p_c_new_list[i,1,:],p_c_new_list[i,2,:])
        p_list.append([x,y,z])
        p_c_list.append([x_c,y_c,z_c])

    p_list = np.asarray(p_list)
    p_c_list = np.asarray(p_c_list)


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

    # run script to create mesh
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

coils = 3  # number of coils
h = coils * 0.0103  # max height
N = 2 * np.pi * coils  # angular turns (radians)
n = 12  # points to use

data = {}
nominal_data = {}


z_vals = np.linspace(0, h, n)
theta_vals = np.linspace(0+np.pi/2, N+np.pi/2, n)
rho_vals = [0.0125 for i in range(n)]
tube_rad_vals = [0.0025 for i in range(n)]
data['fid_radial'] = 4
data['fid_axial'] = 8
for i in range(n):
    nominal_data["z_" + str(i)] = z_vals[i]
    data['z_'+str(i)] = np.random.uniform(-0.002,0.002)
    data['rho_'+str(i)] = np.random.uniform(-0.0075,0.0025)
    # data['rho_'+str(i)] = 0
    # data['z_'+str(i)] = 0
    nominal_data["theta_" + str(i)] = theta_vals[i]
    nominal_data["tube_rad_" + str(i)] = tube_rad_vals[i]
    nominal_data["rho_" + str(i)] = rho_vals[i]
nominal_data['r_c'] = 0.0125

create_mesh(data,'mesh_generation/test',n,nominal_data)
