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

from PIL import Image
import imageio


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

def create_circle(d,radius):
    # from a centre, radius, and z rotation,
    #  create the points of a circle
    c_x, c_y, t, t_x, c_z = d
    angles = np.linspace(0,np.pi*2,len(radius),endpoint=False)
    r_mean = np.mean(radius)
    r_std = np.std(radius)
    if r_std == 0:
        r_std = -r_mean

    radius = (radius - r_mean)/r_std
    angles,radius = gp_interpolate_polar(angles.reshape(-1,1),radius.reshape(-1,1),63)
    radius  = radius * r_std + r_mean
    for i in range(len(radius)):
        if radius[i] != radius[i]:
            radius = radius.at[i].set(r_mean)

    angles = angles[:,0]

    z = radius * np.cos(angles) + c_z
    x = radius * np.sin(angles) + c_x
    y = [c_y for i in range(len(x))]
    x,y,z = rotate_xyz(x,y,z,t,t_x,c_x,c_y,c_z)

    return x, y, z


def cylindrical_convert(r, theta, z):
    # conversion to cylindrical coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = z
    return x, y, z


def interpolate(y, fac_interp, kind):

    x = np.linspace(0, len(y), len(y))
    x_new = np.linspace(0, len(y), len(y) * fac_interp)
    f = interp1d(x, y, kind=kind)
    y_new = f(x_new)
    y = y_new 
    fac = 2
    cutoff = 0.1
    x_start = np.linspace(0,int(len(y)*cutoff),int(len(y)*cutoff))
    x_start_new = np.linspace(0,int(len(y)*cutoff),int(len(y)*cutoff)*fac)
    f = interp1d(x_start, y[:int(len(y)*cutoff)], kind=kind)
    y_start_new = f(x_start_new)
    x_end = np.linspace(0,int(len(y)*cutoff),int(len(y)*cutoff))
    x_end_new = np.linspace(0,int(len(y)*cutoff),int(len(y)*cutoff)*fac)
    f = interp1d(x_end, y[len(y)-int(len(y)*cutoff):], kind=kind)
    y_end_new = f(x_end_new)
    y_new = np.concatenate((y_start_new,y[int(len(y)*cutoff):int(len(y)*(1-cutoff))],y_end_new))

    return y_new

def plot_block(block,ax):
    block = np.array(block)
    lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]

    cols = ['k','tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:gray','tab:brown']
    ax.scatter(block[:,0],block[:,1],block[:,2],c=cols,alpha=1)
    # for i in range(len(lines)):
    #     l = lines[i]
    #     ax.plot([block[l[0],0],block[l[1],0]],[block[l[0],1],block[l[1],1]],[block[l[0],2],block[l[1],2]],c=cols[i],lw=1)
    return 


def create_mesh(interp_points,x: dict, path: str,debug: bool):
    coil_rad = x["coil_rad"]
    pitch = x["pitch"]
    length = x['length']
    r_c = x['radius_center']



    fid_rad = int(x["fid_radial"])
    fid_ax = int(x["fid_axial"])

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
    data["y"] = [(coil_rad * np.sin(x_y)) for x_y in coil_vals]
    # rotations around z are defined by number of coils
    data["t"] = list(coil_vals)
    data["t_x"] = [t_x for i in range(n)]
    # height is linear
    data["z"] = list(np.linspace(0, h , n))
    
    mesh = Mesh()

    fig, axs = plt.subplots(1, 3, figsize=(10, 4), subplot_kw=dict(projection="3d"))
    fig.tight_layout()

    p_list = []
    p_c_list = []
    for i in tqdm(range(n)):
        x, y, z = create_circle(
            [data[keys[j]][i] for j in range(len(keys))],interp_points[i]
        )
        x_c, y_c, z_c = create_center_circle(
            [data[keys[j]][i] for j in range(len(keys))],r_c
        )
        p_list.append([x,y,z])
        p_c_list.append([x_c,y_c,z_c])

    p_list = np.asarray(p_list)
    p_c_list = np.asarray(p_c_list)

    p_new_list= [([interpolate(p_list[:,j,i], fid_ax, "quadratic") for j in range(3)]) for i in range(len(p_list[0,0,:]))]
    p_c_new_list= [([interpolate(p_c_list[:,j,i], fid_ax, "quadratic") for j in range(3)]) for i in range(len(p_c_list[0,0,:]))]

    p_list = np.asarray(p_new_list)
    p_c_list = np.asarray(p_c_new_list)

    # [circle points, coordinates, number of circles]

    if debug == True:
        p_list= p_list[:,:,:2]
        p_c_list= p_c_list[:,:,:2]


    col = (212/255,41/255,144/255)
    for i in range(len(p_list[0,0,:])):

        spacing = (len(p_list[:,0,i])+1)/8
        starts = np.append(np.linspace(0,(len(p_list[:,0,i])+1),4,endpoint=False),0)
        # for seg in range(4): 
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

                for ax in axs:
                    plot_block(block_points,ax)

                cols = ['k','tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:gray','tab:brown']
                spline_indices = np.arange(starts[seg],starts[seg]+spacing)
                spline_1 = [list(p_c_list[int(s),:,i+1]) for s in spline_indices]
                spline_2 = [list(p_c_list[int(s),:,i]) for s in spline_indices]
                spline_indices = np.append(np.arange(starts[seg]+spacing,starts[seg]+spacing+spacing-1),0)
                spline_3 = [list(p_c_list[int(s),:,i+1]) for s in spline_indices]
                spline_4 = [list(p_c_list[int(s),:,i]) for s in spline_indices]
                block_edges = [
                        Edge(4, 5, spline_2), 
                        Edge(5, 6, spline_4),
                        Edge(0, 1, spline_1),
                        Edge(1, 2, spline_3),
                    ]
                
                block = Block.create_from_points(block_points, block_edges)
                block.chop(0,count=3)
                block.chop(1,count=3)
                block.chop(2,count=3)
                mesh.add_block(block)





        #     block.set_patch(["front"], "wall")

        for ax in axs:

            ax.plot3D(p_list[:,0,i], p_list[:,1,i],p_list[:,2,i], color=col, lw=2,zorder=-1)
            ax.plot3D(p_c_list[:,0,i], p_c_list[:,1,i],p_c_list[:,2,i], color=col, lw=2,zorder=-1)

            # if i != len(x_list)-1 and debug != True:
            #         for j in np.linspace(0, len(x_list[i]) - 1, 10):
            #             j = int(j)
            #             ax.plot3D([x_list[i][j], x_list[i+1][j]],[y_list[i][j],y_list[i+1][j]],[z_list[i][j],z_list[i+1][j]], c=col, lw=0.5)

        # calculate x y and z coordinates of inner circle from x y z 
        # coordinates of outer circle
        #     
        


        # div = 8
        # l = np.linspace(0, len(x1), div + 1)[:div].astype(int)
        # l = np.append(l, 0)

        # c1 = np.mean(np.array([[x1[i], y1[i], z1[i]] for i in range(len(x1))]), axis=0)
        # c2 = np.mean(np.array([[x2[i], y2[i], z2[i]] for i in range(len(x1))]), axis=0)

        # fa = 0.8
        # op1 = np.array([[x1[i], y1[i], z1[i]] for i in l])
        # ip1 = np.array([(fa * op1[i] + (1 - fa) * c1) for i in range(len(l))])

        # op2 = np.array([[x2[i], y2[i], z2[i]] for i in l])
        # ip2 = np.array([(fa * op2[i] + (1 - fa) * c2) for i in range(len(l))])

        # for i in range(len(l) - 1):
        #     block_points = [
        #         op1[i],
        #         op1[i + 1],
        #         ip1[i + 1],
        #         ip1[i],
        #     ] + [
        #         op2[i],
        #         op2[i + 1],
        #         ip2[i + 1],
        #         ip2[i],
        #     ]

        #     if i == len(l) - 2:
        #         v = l[i] + (l[i] - l[i - 1])
        #         a = int(l[i] + (v - l[i]) / 2)
        #     else:
        #         a = int(l[i] + (l[i + 1] - l[i]) / 2)

        #     # add circular curves at top of cylindrical quadrant
        #     block_edges = [
        #         Edge(0, 1, [x1[a], y1[a], z1[a]]),  # arc edges
        #         Edge(4, 5, [x2[a], y2[a], z2[a]]),
        #         Edge(2, 3, (fa * np.array([x1[a], y1[a], z1[a]]) + (1 - fa) * c1)),
        #         Edge(6, 7, (fa * np.array([x2[a], y2[a], z2[a]]) + (1 - fa) * c2)),
        #     ]

        #     block = Block.create_from_points(block_points, block_edges)

        #     block.set_patch(["front"], "wall")

        #     # partition block
        #     if p == 1:
        #         block.set_patch("top", "inlet")
        #     if p == le - 3:
        #         block.set_patch("bottom", "outlet")

        #     block.chop(0, count=x["fid_radial"])
        #     block.chop(1, count=x["fid_radial"])
        #     block.chop(2, count=2)

        #     mesh.add_block(block)

        # fa_new = 0.6
        # op1 = ip1
        # ip1 = np.array([(fa_new * op1[i] + (1 - fa_new) * c1) for i in range(len(l))])

        # op2 = ip2
        # ip2 = np.array([(fa_new * op2[i] + (1 - fa_new) * c2) for i in range(len(l))])

        # for i in range(len(l) - 1):
        #     block_points = [
        #         op1[i],
        #         op1[i + 1],
        #         ip1[i + 1],
        #         ip1[i],
        #     ] + [
        #         op2[i],
        #         op2[i + 1],
        #         ip2[i + 1],
        #         ip2[i],
        #     ]

        #     if i == len(l) - 2:
        #         v = l[i] + (l[i] - l[i - 1])
        #         a = int(l[i] + (v - l[i]) / 2)
        #     else:
        #         a = int(l[i] + (l[i + 1] - l[i]) / 2)

        #     # add circular curves at top of cylindrical quadrant
        #     block_edges = [
        #         Edge(
        #             0, 1, (fa * np.array([x1[a], y1[a], z1[a]]) + (1 - fa) * c1)
        #         ),  # arc edges
        #         Edge(4, 5, (fa * np.array([x2[a], y2[a], z2[a]]) + (1 - fa) * c2)),
        #         Edge(2, 3, None),
        #         Edge(6, 7, None),
        #     ]

        #     block = Block.create_from_points(block_points, block_edges)

        #     # block.set_patch(["front"], "wall")

        #     # partition block
        #     if p == 1:
        #         block.set_patch("top", "inlet")
        #     if p == le - 3:
        #         block.set_patch("bottom", "outlet")

        #     block.chop(0, count=x["fid_radial"])
        #     block.chop(1, count=x["fid_radial"])
        #     block.chop(2, count=2)

        #     mesh.add_block(block)

        # i = 1
        # for j in range(int(div / 2)):
        #     # add centre rectangular block
        #     block_points = [ip1[i - 1], ip1[i], ip1[i + 1], c1] + [
        #         ip2[i - 1],
        #         ip2[i],
        #         ip2[i + 1],
        #         c2,
        #     ]
        #     i += 2
        #     block_edges = [
        #         Edge(0, 1, None),  # arc edges
        #         Edge(4, 5, None),
        #         Edge(2, 3, None),  # spline edges
        #         Edge(6, 7, None),
        #     ]
        #     block = Block.create_from_points(block_points, block_edges)
        #     # partition block
        #     if p == 1:
        #         block.set_patch("top", "inlet")
        #     if p == le - 3:
        #         block.set_patch("bottom", "outlet")

        #     block.chop(0, count=x["fid_radial"])
        #     block.chop(1, count=x["fid_radial"])
        #     block.chop(2, count=2)

        #     mesh.add_block(block)
        # # copy template folder

    # plt.show()

    for ax in axs:
        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
        )
        ax.set_axis_off()


    axs[0].view_init(30, 310)
    axs[1].view_init(0, 210)
    axs[2].view_init(240, 0)


    try:
        shutil.copytree("mesh_generation/mesh", path)
    except FileExistsError:
        print("Folder already exists")

    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.05, top=0.99, bottom=0.01)
    plt.savefig(path+'pre_render.png', dpi=600)

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
#     create_mesh(combo,coil_data,'mesh_generation/test/'+str(i)+'.png')



# images = [] # creating image array
# for i in range(30): # iterating over images
#     im = Image.open('mesh_generation/test/'+str(i)+'.png')
#     images.append(im)
#     os.remove('mesh_generation/test/'+str(i)+'.png') # this then deletes the image file from the folder
# imageio.mimsave('mesh_generation/test/res.gif', images, format='GIF-PIL', quantizer=0) # this then saves the array of images as a gif


# -----------------------------

n = 8
n_cross_section = 8
coil_data = {"radius_center":0.001,"length":0.0753,"pitch":0.0075,"coil_rad":0.0085,"fid_axial":10,"fid_radial":2}

cross_section_points = [np.random.uniform(0.0015,0.0035,n_cross_section) for i in range(n)]
# cross_section = [np.array([0.0025 for i in range(n_cross_section)]) for i in range(n)]
create_mesh(cross_section_points,coil_data,'mesh_generation/test/',debug=True)

