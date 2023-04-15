import numpy as np
from scipy.interpolate import interp1d
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
sys.path.insert(1, "mesh_generation/classy_blocks/src/")

from classy_blocks.classes.primitives import Edge
from classy_blocks.classes.block import Block
from jax.nn import softplus
from gpjax.config import add_parameter
from classy_blocks.classes.mesh import Mesh
import shutil
import matplotlib.pyplot as plt
import math
from matplotlib import rc
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


def rotate_z(x0, y0, z0, r_z):
        # rotation of points around z axis by r_z radians
        x = np.array(x0) - np.mean(x0)
        y = np.array(y0) - np.mean(y0)
        x_new = x * np.cos(r_z) - y * np.sin(r_z)
        y_new = x * np.sin(r_z) + y * np.cos(r_z)
        x_new += np.mean(x0)
        y_new += np.mean(y0)
        return x_new, y_new, z0



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



def unit_vector(vector):
        # returns a unit vector
        return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
        # angle between two vectors
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def calc_angle(d1, d2):
        # takes 2 cylindrical coordinates
        # and rotates the location of the second to be orthogonal
        # to the vector between the centre of the two circles

        # circle_test.py provides an example
        r1, t1, z1, rad1 = d1
        r2, t2, z2, rad2 = d2
        x1,y1,z1 = cylindrical_convert(r1, t1, z1)
        x2,y2,z2 = cylindrical_convert(r2, t2, z2)
        c1 = np.array([x1, y1, z1])
        c2 = np.array([x2, y2, z2])
        v = c2 - c1
        a_x = angle_between([0, 0, 1], v)
        xv = [v[0], v[1]]
        b = [0, 1]
        a_z = math.atan2(xv[1] * b[0] - xv[0] * b[1], xv[0] * b[0] + xv[1] * b[1])
        return -a_x,a_z

def create_circle_known(d2,r_x,r_z):
        r2, t2, z2, rad2 = d2
        c_x2, c_y2, c_z2 = cylindrical_convert(r2, t2, z2)
        alpha = np.linspace(0, 2 * np.pi, 64)
        y2 = 2*rad2 * np.cos(alpha) + c_y2
        x2 = 2*rad2 * np.sin(alpha) + c_x2
        z2 = [c_z2 for i in range(len(x2))]
        c2 = np.mean([x2, y2, z2], axis=1)
        x2p, y2p, z2p = rotate_x(x2, y2, z2, r_x)
        x2p, y2p, z2p = rotate_z(x2p, y2p, z2p, r_z)

        return x2p + c2[0], y2p + c2[1], z2p + c2[2]

def interpolate(y, fac_interp, kind, name):

        x = np.linspace(0, len(y), len(y))
        x_new = np.linspace(0, len(y), len(y) * fac_interp)
        if len(y) == 2:
                kind = 'linear'
        f = interp1d(x, y, kind=kind)
        y_new = f(x_new)
        y = y_new 

        return y_new,x_new

def interpolate_num(y, num, kind):
        x = np.linspace(0, len(y), len(y))
        x_new = np.linspace(0, len(y), num)
        if len(y) == 2:
                kind = 'linear'
        f = interp1d(x, y, kind=kind)
        y_new = f(x_new)
        y = y_new 
        return y_new,x_new

def plot_block(block,ax):
        block = np.array(block)
        lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
        cols = ['k','tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:gray','tab:brown']
        # ax.scatter(block[:,0],block[:,1],block[:,2],c='colks',alpha=1)
        for i in range(len(lines)):
                l = lines[i]
                ax.plot([block[l[0],0],block[l[1],0]],[block[l[0],1],block[l[1],1]],[block[l[0],2],block[l[1],2]],c='tab:blue',lw=2,alpha=0.25)
        return 
def interpolate_num_same(x,x_new,y, kind):
        f = interp1d(x, y, kind=kind)
        y_new = f(x_new)
        y = y_new 
        return y_new,x_new

def parse_inputs(NB, f, name):
        y,x = interpolate(NB, f, "quadratic", name)
        return y,x

def smooth_path(start,end,x,d):
        x_slice = x[start:end]
        x_m = (x_slice[0] + x_slice[-1])/2
        w = 2
        x_h = x_slice[int(len(x_slice)/2)]
        w_x_m = (w*x_h +x_m)/(w+1)
        x_interp = [x_slice[0],w_x_m,x_slice[-1]]
        x_new,_ = interpolate_num_same(x_interp,x_slice, len(x_slice), "quadratic")
        insert = list(range(start,end))
        for i in range(len(insert)):
                x[insert[i]] = x_new[i]
        return x

def interpolate_path(r,theta,z,f):

        x,y,z = cylindrical_convert(r,theta,z)
        rho_mid = interpolate_s(r[1:], f, "quadratic")
        theta_mid = interpolate_s(theta[1:], f, "quadratic")
        z_mid = interpolate_s(z[1:], f, "quadratic")
        x_m,y_m,z_m = cylindrical_convert(rho_mid,theta_mid,z_mid)

        x_start = interpolate_s([x[0],x[1]], int(f/2), "linear")
        y_start = interpolate_s([y[0],y[1]], int(f/2), "linear")
        z_start = interpolate_s([z[0],z[1]], int(f/2), "linear")
        x_start = x_start[:-1]
        y_start = y_start[:-1]
        z_start = z_start[:-1]

        d_start = np.sqrt((x_start[0]-x_start[-1])**2 + (y_start[0]-y_start[-1])**2 + (z_start[0]-z_start[-1])**2)

        x = np.append(x_start,x_m)
        y = np.append(y_start,y_m)
        z = np.append(z_start,z_m)

        len_s = len(x_start)

        d_store = [0]
        for i in range(1,len(x)):
                d = np.sqrt((x[i-1]-x[i])**2 + (y[i-1]-y[i])**2 + (z[i-1]-z[i])**2)
                d_store.append(d)
        d_store = np.cumsum(d_store)

        s = len_s - int(f/2) 
        e = len_s + int(f/2)

        d_int = [d_store[s+0],d_store[s+1],d_store[e-2],d_store[e-1]]
        z_int = [z[s+0],z[s+1],z[e-2],z[e-1]]

        z_new,_ = interpolate_num_same(d_int,d_store[s:e],z_int,"quadratic")


        for i in range(s,e):
                z[i] = z_new[i-s]

        rho,theta,z = cartesian_convert(x,y,z)
        return rho,theta,z

# start do from parralel
# end do from dx dy method 

def interpolate_s(y, fac_interp, kind):

        x = np.linspace(0, len(y), len(y))
        x_new = np.linspace(0, len(y), len(y) * fac_interp)
        f = interp1d(x, y, kind=kind)
        y_new = f(x_new)
        y = y_new 

        return y

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



def create_gp_circle(d,radius_og,rot_x,rot_z):
        # from a centre, radius, and z rotation,
        #  create the points of a circle
        c_x, c_y, c_z = d
        radius = radius_og.copy()
        angles_og = np.linspace(0,np.pi*2,len(radius),endpoint=False)
        angles = angles_og.copy()
        r_mean = np.mean(radius)
        r_std = np.std(radius)
        if r_std != 0:
                radius = (radius - r_mean)/r_std
                angles,radius = gp_interpolate_polar(angles.reshape(-1,1),radius.reshape(-1,1),64)
                angles = angles[:,0]
                radius  = radius * r_std + r_mean
        else:
                angles = np.linspace(0,np.pi*2,64)
                radius = np.array([r_mean for i in range(64)])

        for i in range(len(radius)):
                if radius[i] != radius[i]:
                        radius = radius.at[i].set(r_mean)

        y_n = radius * np.cos(angles) +c_y
        x_n = radius * np.sin(angles) + c_x
        z_n =  np.array([c_z for i in range(len(x_n))])
        x, y, z = rotate_x(x_n, y_n, z_n, rot_x)
        x, y, z = rotate_z(x, y, z, rot_z)

        return x, y, z


def add_start(rho,theta,z,L):
        rho_start = np.sqrt(rho[0]**2 + L**2)
        theta_start = theta[0] - np.arctan(L/rho[0])
        z_start = z[0]

        z = np.append(z_start,z)
        rho = np.append(rho_start,rho)
        theta = np.append(theta_start,theta)

        return rho,theta,z

def interpolate_num_same(x,x_new,y, kind):
        f = interp1d(x, y, kind=kind)
        y_new = f(x_new)
        y = y_new 
        return y_new,x_new

def interpolate_split(r,theta,z,fid_ax):
        #z[1] = (z[0]+z[2])/2
        x,y,z = cylindrical_convert(r,theta,z)
        rho_mid = interpolate_s(r[1:], fid_ax, "quadratic")
        theta_mid = interpolate_s(theta[1:], fid_ax, "quadratic")
        z_mid = interpolate_s(z[1:], fid_ax, "quadratic")
        x_m,y_m,z_m = cylindrical_convert(rho_mid,theta_mid,z_mid)

        x_start = interpolate_s([x[0],x[1]], int(fid_ax/2), "linear")
        y_start = interpolate_s([y[0],y[1]], int(fid_ax/2), "linear")
        z_start = interpolate_s([z[0],z[1]], int(fid_ax/2), "linear")
        x_start = x_start[:-1]
        y_start = y_start[:-1]
        z_start = z_start[:-1]

        d_start = np.sqrt((x_start[0]-x_start[-1])**2 + (y_start[0]-y_start[-1])**2 + (z_start[0]-z_start[-1])**2)

        x = np.append(x_start,x_m)
        y = np.append(y_start,y_m)
        z = np.append(z_start,z_m)

        len_s = len(x_start)

        d_store = [0]
        for i in range(1,len(x)):
                d = np.sqrt((x[i-1]-x[i])**2 + (y[i-1]-y[i])**2 + (z[i-1]-z[i])**2)
                d_store.append(d)
        d_store = np.cumsum(d_store)

        s = len_s - int(fid_ax/2) 
        e = len_s + int(fid_ax/2)

        d_int = [d_store[s+0],d_store[s+1],d_store[e-2],d_store[e-1]]


        z_int = [z[s+0],z[s+1],z[e-2],z[e-1]]
        x_int = [x[s+0],x[s+1],x[e-2],x[e-1]]
        y_int = [y[s+0],y[s+1],y[e-2],y[e-1]]

        
        z_new,_ = interpolate_num_same(d_int,d_store[s:e],z_int,"quadratic")
        x_new,_ = interpolate_num_same(d_int,d_store[s:e],x_int,"quadratic")
        y_new,_ = interpolate_num_same(d_int,d_store[s:e],y_int,"quadratic")


        for i in range(s,e):
                z[i] = z_new[i-s]
                x[i] = x_new[i-s]
                y[i] = y_new[i-s]

        # r = np.append(r_start,r)
        # theta = np.append(theta_start,theta)
        # z_c = np.append(z_start,z_c)

        return x,y,z,d_start

def add_end(x,y,z,dx,dy,dz,d_start,fid_ax):

        d_end = np.sqrt(dx**2+dy**2+dz**2)
        factor = d_start / d_end
        
        x_e = x[-1] + dx * factor
        y_e = y[-1] + dy * factor
        z_e = z[-1] + dz * factor

        x_end = interpolate_s([x[-1],x_e], int(fid_ax/2), "linear")
        y_end = interpolate_s([y[-1],y_e], int(fid_ax/2), "linear")
        z_end = interpolate_s([z[-1],z_e], int(fid_ax/2), "linear")

        x_end = x_end[1:]
        y_end = y_end[1:]
        z_end = z_end[1:]

        x = np.append(x,x_end)
        y = np.append(y,y_end)
        z = np.append(z,z_end)

        return x,y,z


def create_mesh(data,interp_points, path, n_interp, nominal_data_og):

        nominal_data = nominal_data_og.copy()
        try:
                shutil.copytree("mesh_generation/mesh", path)
        except:
                print('file already exists')
        # factor to interpolate between control points

        interp_points.append(np.array([nominal_data['tube_rad_0'] for i in range(len(interp_points[0]))])) # adding start and end inlet to be correct
        interp_points.insert(0,np.array([nominal_data['tube_rad_0'] for i in range(len(interp_points[0]))]))
        interp_points.insert(0,np.array([nominal_data['tube_rad_0'] for i in range(len(interp_points[0]))]))

        interpolation_factor =int((data["fid_axial"]))
        fid_ax = int((data["fid_axial"]))
        # interpolate x times the points between
        fid_radial = int((data["fid_radial"]))
        fid_rad = int((data["fid_radial"]))

        keys = ["rho", "theta", "z", "tube_rad"]

        # do interpolation between points
        vals = {}

        for i in range(6):
                try:
                        nominal_data["rho_" + str(i)] += data["rho_" + str(i)]
                except:
                        nominal_data["rho_" + str(i)] += 0 
                try:
                        nominal_data["z_" + str(i)] += data["z_" + str(i)]
                except:
                        nominal_data["z_" + str(i)] += 0

        data = nominal_data
        data_og = nominal_data_og 


        rho = data["rho_"+str(n_interp-1)]
        theta = data["theta_"+str(n_interp-1)]


        nominal_data = nominal_data_og.copy()
        vals_og = {}

        vals = {}
        for k in keys:
                vals[k] = [data[k + "_" + str(i)] for i in list(range(n_interp))]

        data = vals 


        x,y,z = cylindrical_convert(vals['rho'],vals['theta'],vals['z'])

        

        y = -y 
        vals['rho'],vals['theta'],vals['z'] = cartesian_convert(x,y,z)
        vals['rho'],vals['theta'],vals['z'] = add_start(vals['rho'],vals['theta'],vals['z'],rho)
        vals['rho'] = vals['rho'][:-1]
        vals['theta'] = vals['theta'][:-1]
        vals['z'] = vals['z'][:-1]
        data['rho'],data['theta'],data['z'] = interpolate_path(vals['rho'],vals['theta'],vals['z'],interpolation_factor)
        # x,y,z = cylindrical_convert(vals['rho'],vals['theta'],vals['z'])
        # fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})
        # ax.plot(x,y,z)
        # plt.show()
        data['tube_rad'] = [nominal_data['tube_rad_0'] for i in range(len(data['rho']))]
        le = len(data['z'])

        x,y,z = cylindrical_convert(data['rho'],data['theta'],data['z'])
        # fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})
        # ax.plot(x,y,z)
        # plt.show()


        data["fid_radial"] = fid_radial

        mesh = Mesh()

        rot_x_store = []
        rot_z_store = []

        for p in range(1, le ):
                # get proceeding circle (as x,y,z samples)
                rot_x,rot_z = calc_angle(
                        [data[keys[i]][p - 1] for i in range(len(keys))],
                        [data[keys[i]][p] for i in range(len(keys))]
                )

                rot_x_store.append(rot_x)
                rot_z_store.append(rot_z)

        for i in range(1,len(rot_z_store)):
                if rot_z_store[i] + 1 < rot_z_store[i-1]:
                        for j in range(i,len(rot_z_store)):
                                rot_z_store[j] += 2*np.pi

        
        x,y,z = cylindrical_convert(data['rho'],data['theta'],data['z'])

        s_ind = np.linspace(1,len(data['rho'])-1,len(interp_points),dtype=int)
        data['rho'] = [data['rho'][i] for i in s_ind]
        data['theta'] = [data['theta'][i] for i in s_ind]
        data['z'] = [data['z'][i] for i in s_ind]


        rot_x_store = [rot_x_store[i-1] for i in s_ind]
        rot_z_store = [rot_z_store[i-1] for i in s_ind]


        data['tube_rad'] = [nominal_data['tube_rad_0'] for i in range(len(data['rho']))]
        # rot_x_store = [0] + rot_x_store
        # rot_z_store = [0] + rot_z_store
        # rot_x_store[1] = 0
        # rot_z_store[1] = 0

        data['x'],data['y'],data['z'] = cylindrical_convert(data['rho'],data['theta'],data['z'])

        n = len(data['x'])
        p_list = []
        keys = ['x','y','z']
        for i in range(n):
                x, y, z = create_gp_circle(
                        [data[keys[j]][i] for j in range(len(keys))],interp_points[i],rot_x_store[i],rot_z_store[i]
                )
                
                p_list.append([x,y,z])

        fig_p, axs_p = plt.subplots(1, 3, figsize=(10, 3), subplot_kw=dict(projection="3d"))
        le = len(rot_x_store)
  

        p_list = np.asarray(np.array(p_list))

        p_cylindrical_list = []

        for i in range(len(p_list[0,0,:])):
                r,theta,z = cartesian_convert(p_list[:,0,i],p_list[:,1,i],p_list[:,2,i])
                p_cylindrical_list.append([r,theta,z])

        p_cylindrical_list = np.asarray(p_cylindrical_list)
        p_new_list = [] 
        for i in range(len(p_cylindrical_list[:,0,0])):

                x,y,z,d_start = interpolate_split(p_cylindrical_list[i,0,:],p_cylindrical_list[i,1,:],p_cylindrical_list[i,2,:], fid_ax)
                p_new_list.append([x,y,z])

        p_new_list = np.asarray(p_new_list)

        m_x = np.mean(p_new_list[:,0,:],axis=0)
        m_y = np.mean(p_new_list[:,1,:],axis=0)
        m_z = np.mean(p_new_list[:,2,:],axis=0)
        dx = m_x[-1] - m_x[-2]
        dy = m_y[-1] - m_y[-2]
        dz = m_z[-1] - m_z[-2]

        p_list = []
        for i in range(len(p_new_list[:,0,0])):
                x,y,z = add_end(p_new_list[i,0,:],p_new_list[i,1,:],p_new_list[i,2,:], dx,dy,dz,d_start,fid_ax)
                p_list.append([x,y,z])

        p_list = np.asarray(p_list)

        # plot for reference
        col = 'k'
        for ax in axs_p:
                for i in range(len(p_list[:,0,0])-1):
                        ax.plot3D(p_list[i,0,:],p_list[i,1,:],p_list[i,2,:], c=col, lw=0.5,alpha=0.25)
                for j in range(len(p_list[0,0,:])-1):
                        ax.plot3D(p_list[:,0,j],p_list[:,1,j],p_list[:,2,j],c=col,lw=0.5,alpha=0.25)

        axs_p[0].view_init(0, 270)
        axs_p[1].view_init(0, 180)
        axs_p[2].view_init(270, 0)

        for ax in axs_p:
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
        axs_p[0].set_xlabel("x",fontsize=14)
        axs_p[0].set_zlabel("z",fontsize=14)
        axs_p[1].set_ylabel("y",fontsize=14)
        axs_p[1].set_zlabel("z",fontsize=14)
        axs_p[2].set_ylabel("y",fontsize=14)
        axs_p[2].set_xlabel("x",fontsize=14)

        fig_p.tight_layout()
        fig_p.savefig(path + "/pre-render.png", dpi=600)
        p_list = p_list.T
        c = np.mean(p_list,axis=2)
        s = 0 
        ds = int((len(p_list[0,0,:]))/4)
        c_same = np.zeros_like(p_list)
        for i in range(len(p_list[0,0,:])):
                c_same[:,:,i] = c
        inner_p_list = 0.8*p_list + 0.2*c_same
        for i in range(4):
                e = s+ds+1
                quart = p_list[:,:,s:e]
                inner_quart = inner_p_list[:,:,s:e]


                c_weight = 0.4
                mid_ind = int(len(quart[0,0,:])/2)

                p_30 = c
                p_74 = (c_weight*quart[:,:,0]) + ((1-c_weight)*p_30)
                p_21 = (c_weight*quart[:,:,-1]) + ((1-c_weight)*p_30)
                p_65 = 0.8*(0.5*p_74+0.5*p_21) + 0.2*quart[:,:,mid_ind]
                for k in range(len(quart[:,0,0])-1):

                        block_points = [list(p_30[k+1,:]),list(p_21[k+1,:]),list(p_21[k,:]),list(p_30[k,:]),list(p_74[k+1,:]),list(p_65[k+1,:]),list(p_65[k,:]),list(p_74[k,:])]
                        block_edges = []
                        
                        block = Block.create_from_points(block_points, block_edges)
                        # block.set_patch("top", "inlet")
                        # block.set_patch("bottom", "outlet")

                        block.chop(0, count = data['fid_radial'])
                        block.chop(1, count = 1)
                        block.chop(2, count = data['fid_radial'])

                        mesh.add_block(block)
                        if k == 0:
                                block.set_patch("back", "inlet")
                        if k == len(quart[:,0,0])-2:
                                block.set_patch("front", "outlet")

                
                for m_in in [0,int(len(quart[0,0,:])/2)]:


                        m_out = m_in + mid_ind
                        if m_out == len(quart[0,0,:]):
                                m_out -= 1
                        p_74_ = inner_quart[:,:,m_in]
                        p_74_u = quart[:,:,m_in]
                        p_65_ = inner_quart[:,:,m_out]
                        p_65_u = quart[:,:,m_out]
                        p_30_u = p_74_
                        p_21_u = p_65_
                        if m_in == 0:
                                p_30_ = p_74
                                p_21_ = p_65

                        else:
                                p_30_ = p_65
                                p_21_ = p_21


                        for k in range(len(quart[:,0,0])-1):
                                curve_76 = inner_quart[0,:,m_in:m_out].T
                                curve_76_u = quart[0,:,m_in:m_out].T
                                curve_76 = list([list(curve_76[int(i),:]) for i in np.linspace(0,len(curve_76[:,0])-1,fid_rad)])
                                curve_76_u = list([list(curve_76_u[int(i),:]) for i in np.linspace(0,len(curve_76_u[:,0])-1,fid_rad)])
                                curve_45 = inner_quart[k+1,:,m_in:m_out].T
                                curve_45_u = quart[k+1,:,m_in:m_out].T
                                curve_45 = list([list(curve_45[int(i),:]) for i in np.linspace(0,len(curve_45[:,0])-1,fid_rad)])
                                curve_45_u = list([list(curve_45_u[int(i),:]) for i in np.linspace(0,len(curve_45_u[:,0])-1,fid_rad)])

                                block_points = [list(p_30_[k+1,:]),list(p_21_[k+1,:]),list(p_21_[k,:]),list(p_30_[k,:]),list(p_74_[k+1,:]),list(p_65_[k+1,:]),list(p_65_[k,:]),list(p_74_[k,:])]
                                block_edges = [
                                                Edge(7,6, curve_76),
                                                Edge(4,5, curve_45)
                                        ]
                                
                                block = Block.create_from_points(block_points, block_edges)
                                block.chop(0, count = data['fid_radial'])
                                block.chop(1, count = 1)
                                block.chop(2, count = data['fid_radial'])
                                if k == 0:
                                        block.set_patch("back", "inlet")
                                if k == len(quart[:,0,0])-2:
                                        block.set_patch("front", "outlet")
                                mesh.add_block(block)

                                block_points = [list(p_30_u[k+1,:]),list(p_21_u[k+1,:]),list(p_21_u[k,:]),list(p_30_u[k,:]),list(p_74_u[k+1,:]),list(p_65_u[k+1,:]),list(p_65_u[k,:]),list(p_74_u[k,:])]
                                block_edges = [
                                                Edge(7,6, curve_76_u),
                                                Edge(4,5, curve_45_u),
                                                Edge(0,1,curve_45),
                                                Edge(3,2,curve_76)
                                        ]
                                block = Block.create_from_points(block_points, block_edges)

                                block.chop(0, count = data['fid_radial'])
                                block.chop(1, count = 1)
                                block.chop(2, count = data['fid_radial'])
                                if k == 0:
                                        block.set_patch("back", "inlet")
                                if k == len(quart[:,0,0])-2:
                                        block.set_patch("front", "outlet")
                                block.set_patch("top", "wall")
                                mesh.add_block(block)
                s += ds

        # # run script to create mesh
        print("Writing geometry")
        mesh.write(output_path=os.path.join(path, "system", "blockMeshDict"), geometry=None)
        print("Running blockMesh")
        os.system("chmod +x " + path + "/Allrun.mesh")
        os.system(path + "/Allrun.mesh")

        return


# coils = 2  # number of coils
# h = coils * 0.010391  # max height
# N = 2 * np.pi * coils  # angular turns (radians)
# n = 8  # points to use

# data = {}
# nominal_data = {}

# data['fid_radial'] = 4
# data['fid_axial'] = 40

# data['rho_0'] = 0
# data['z_1'] = 0
# data['z_0'] = 0
# data['rho_1'] = 0
# for i in range(2,n):
#         # data['z_'+str(i)] = 0
#         # data['rho_'+str(i)] = 0
#         data['z_'+str(i)] = np.random.uniform(-0.001,0.001)
#         data['rho_'+str(i)] = np.random.uniform(-0.0025,0.001)

# z_vals = np.linspace(0, h/2, n)
# theta_vals = np.flip(np.linspace(0+np.pi/2, N+np.pi/2, n))
# rho_vals = [0.0125/2 for i in range(n)]
# tube_rad_vals = [0.0025/2 for i in range(n)]
# for i in range(n):
#         nominal_data["z_" + str(i)] = z_vals[i]
#         nominal_data["theta_" + str(i)] = theta_vals[i]
#         nominal_data["tube_rad_" + str(i)] = tube_rad_vals[i]
#         nominal_data["rho_" + str(i)] = rho_vals[i]


# n_circ = 6
# n_cross_section = 6


# x_list = []
# for i in range(n_circ):
#         x_add = []
#         for j in range(n_cross_section):
#                 x_add.append(np.random.uniform(0.001,0.002))
#                 # x_add.append(0.0025)
#         x_list.append(np.array(x_add))

# create_mesh(data,x_list,'mesh_generation/test',n,nominal_data)