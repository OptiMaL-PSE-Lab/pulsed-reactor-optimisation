import numpy as np
from scipy.interpolate import interp1d
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
sys.path.insert(1, "mesh_generation/classy_blocks/src/")

from classy_blocks.classes.primitives import Edge
from classy_blocks.classes.block import Block
from classy_blocks.classes.mesh import Mesh
import shutil
import matplotlib.pyplot as plt
import math
from matplotlib import rc


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

def interpolate_path(rho,theta,z,f):

        x,y,z = cylindrical_convert(rho,theta,z)
        x_start,_ = interpolate([x[0],x[1]], int(f/2), "linear", "x")
        y_start,_ = interpolate([y[0],y[1]], int(f/2), "linear", "y")
        z_start,_ = interpolate([z[0],z[1]], int(f/2), "linear", "z")


        d_start = np.sqrt((x_start[0]-x_start[-1])**2 + (y_start[0]-y_start[-1])**2 + (z_start[0]-z_start[-1])**2)

        rho_mid,_ = parse_inputs(rho[1:-1], f, "rho")
        theta_mid,_ = parse_inputs(theta[1:-1], f, "theta")
        z_mid,_ = parse_inputs(z[1:-1], f, "z")
        x_m,y_m,z_m = cylindrical_convert(rho_mid,theta_mid,z_mid)
        dx = x_m[-1] - x_m[-2]
        dy = y_m[-1] - y_m[-2]
        dz = z_m[-1] - z_m[-2]


        d_end = np.sqrt(dx**2+dy**2+dz**2)
        factor = d_start / d_end
        
        x_e = x_m[-1] + dx * factor
        y_e = y_m[-1] + dy * factor
        z_e = z_m[-1] + dz * factor

        x_end,_ = interpolate([x_m[-1],x_e], int(f/2), "linear", "x")
        y_end,_ = interpolate([y_m[-1],y_e], int(f/2), "linear", "y")
        z_end,_ = interpolate([z_m[-1],z_e], int(f/2), "linear", "z")

        x_end = x_end[1:]
        y_end = y_end[1:]
        z_end = z_end[1:]



        rho_start,theta_start,z_start = cartesian_convert(x_start,y_start,z_start)
        rho_end,theta_end,z_end = cartesian_convert(x_end,y_end,z_end)


        rho = np.append(np.append(rho_start,rho_mid[1:]),rho_end)
        theta = np.append(np.append(theta_start,theta_mid[1:]),theta_end)
        z = np.append(np.append(z_start,z_mid[1:]),z_end)


        x,y,z = cylindrical_convert(rho,theta,z)
        len_s = len(rho_start) 
        y = -y 

        rho,theta,z = cartesian_convert(x,y,z)
        

        return rho,theta,z,len(x_start),len(z_mid)

# start do from parralel
# end do from dx dy method 

def create_mesh(data, path, n_interp, nominal_data_og):

        try:
                shutil.copytree("mesh_generation/mesh", path)
        except:
                print('file already exists')
        # factor to interpolate between control points
        interpolation_factor =int((data["fid_axial"]))
        # interpolate x times the points between
        fid_radial = int((data["fid_radial"]))
        fid_rad = int((data["fid_radial"]))
        fid_rad = int((data["fid_radial"]))

        keys = ["rho", "theta", "z", "tube_rad"]

        # do interpolation between points
        vals = {}



        fig = plt.figure(figsize=(12,7))
        widths=[1,1,1]
        heights=[1,7]
        fig.add_gridspec(ncols=3, nrows=2, width_ratios=widths,
                          height_ratios=heights)
        axs = []
        for i in range(3):
                axs.append(fig.add_subplot(2, 3, i+1))

        for i in range(3):
                axs.append(fig.add_subplot(2, 3, i+4, projection='3d'))
        for ax in axs[3:]:
                ax.axes.set_facecolor('w')
                ax.axes.set_alpha(0)
        
        fig.tight_layout()
        plt.subplots_adjust(left=0.075,bottom=0.05,wspace=0.3,hspace=0.45)
        x_ax = np.linspace(0,1,n_interp-1) * 100

        for ax in axs[:3]:
                ax.set_xlabel('Coil Length (%)',fontsize=14)
        
        axs[0].set_ylabel('z',fontsize=14)
        axs[1].set_ylabel(r'$\rho$',fontsize=14)
        axs[2].set_ylabel(r'$\theta$',fontsize=14)

        nominal_data = nominal_data_og.copy()

        axs[0].plot(x_ax,[nominal_data['z_'+str(i)] for i in range(n_interp-1)],c='k',lw=2,ls=':')
        axs[1].plot(x_ax,[nominal_data['rho_'+str(i)] for i in range(n_interp-1)],c='k',lw=2,ls=':')
        axs[2].plot(x_ax,[nominal_data['theta_'+str(i)] for i in range(n_interp-1)],c='k',lw=2,ls=':')


        axs[0].scatter(x_ax,[nominal_data['z_'+str(i)]+data['z_'+str(i)] for i in range(n_interp-1)],c='k',lw=2)
        axs[1].scatter(x_ax,[nominal_data['rho_'+str(i)]+data['rho_'+str(i)] for i in range(n_interp-1)],c='k',lw=2)
        for i in range(n_interp-1):
                if i == 0:
                        axs[0].plot([x_ax[i],x_ax[i]],[nominal_data['z_'+str(i)],nominal_data['z_'+str(i)]+data['z_'+str(i)]],c='tab:red',lw=2,label='Parameter (Path Difference)')
                else:
                        axs[0].plot([x_ax[i],x_ax[i]],[nominal_data['z_'+str(i)],nominal_data['z_'+str(i)]+data['z_'+str(i)]],c='tab:red',lw=2)
                axs[1].plot([x_ax[i],x_ax[i]],[nominal_data['rho_'+str(i)],nominal_data['rho_'+str(i)]+data['rho_'+str(i)]],c='tab:red',lw=2)

        axs[0].legend(loc='upper center', bbox_to_anchor=(2.7, -0.3),frameon=False,ncol=2,fontsize=14)

        for i in range(n_interp):
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


        # calculating real values from differences and initial conditions
        rho = data["rho_0"]
        theta = data["theta_0"]

        L = rho 
        rho_two = np.sqrt(rho**2+L**2)
        theta_two = theta + np.arctan(L/rho)

        data["theta_s"] = theta_two
        data_og["theta_s"] = theta_two
        data["rho_s"] = rho_two
        data_og["rho_s"] = rho_two
        data["z_s"] = data["z_0"]
        data_og["z_s"] = data_og["z_0"]
        data['tube_rad_s'] = data['tube_rad_0']
        data_og['tube_rad_s'] = data_og['tube_rad_0']

        rho = data["rho_"+str(n_interp-1)]
        theta = data["theta_"+str(n_interp-1)]


        nominal_data = nominal_data_og.copy()
        vals_og = {}

        vals = {}
        for k in keys:
                vals[k] = [data[k+"_s"]]+[data[k + "_" + str(i)] for i in list(range(n_interp))]
                vals_og[k] = [data_og[k+"_s"]]+[data_og[k + "_" + str(i)] for i in list(range(n_interp))]
        data = {}
        data_og = {}
        data['rho'],data['theta'],data['z'],len_s,len_mid = interpolate_path(vals['rho'],vals['theta'],vals['z'],interpolation_factor)
        data_og['rho'],data_og['theta'],data_og['z'],len_s,len_mid = interpolate_path(vals_og['rho'],vals_og['theta'],vals_og['z'],interpolation_factor)
        data['tube_rad'] = [nominal_data['tube_rad_0'] for i in range(len(data['rho']))]
        data_og['tube_rad'] = [nominal_data_og['tube_rad_0'] for i in range(len(data_og['rho']))]
        # for k in keys:
        #     vals[k],x_ax = parse_inputs(data[k], interpolation_factor, k)
        #vals = data.copy()
        x,y,z = cylindrical_convert(data['rho'],data['theta'],data['z'])


        x_ax = 100*np.arange(len(data['z'])-2*len_s)/(len(data['z'])-2*len_s)
        
        axs[0].plot(x_ax,data['z'][len_s:-len_s],c='k',lw=2)
        axs[1].plot(x_ax,data['rho'][len_s:-len_s],c='k',lw=2)


        x,y,z = cylindrical_convert(data_og['rho'],data_og['theta'],data_og['z'])
        for ax in axs[3:]:
            ax.plot(x,y,z,c='k',lw=2,ls=':',label='Nominal Path')

        x,y,z = cylindrical_convert(data['rho'],data['theta'],data['z'])
        for ax in axs[3:]:
            ax.plot(x,y,z,c='k',lw=2,label='Deformed Path')

        # add legend to bottom of overall figure
        axs[3].legend(loc='upper center', bbox_to_anchor=(1.3, 1.14),frameon=False,ncol=2,fontsize=14)

        for ax in axs[3:]:
                ax.set_box_aspect(
                        [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        axs[3].set_xlabel("x",fontsize=14)
        axs[3].set_zlabel("z",fontsize=14)

        axs[4].set_ylabel("y",fontsize=14)
        axs[4].set_zlabel("z",fontsize=14)

        axs[5].set_ylabel("y",fontsize=14)
        axs[5].set_xlabel("x",fontsize=14)

        axs[3].view_init(0, 270)
        axs[4].view_init(0, 180)
        axs[5].view_init(270, 0)


        plt.savefig(path+'/interp.pdf')
        le = len(data['z'])

        data["fid_radial"] = fid_radial
        mesh = Mesh()

        rot_x_store = []
        rot_z_store = []

        d_store = [0]
        for i in range(1,len(x)):
                d = np.sqrt((x[i-1]-x[i])**2 + (y[i-1]-y[i])**2 + (z[i-1]-z[i])**2)
                d_store.append(d)
        d_store = np.cumsum(d_store)

        s = len_s - int(interpolation_factor/2) 
        e = len_s + int(interpolation_factor/2)

        d_int = [d_store[s+0],d_store[s+1],d_store[e-2],d_store[e-1]]
        z_int = [z[s+0],z[s+1],z[e-2],z[e-1]]
        
        z_new,_ = interpolate_num_same(d_int,d_store[s:e],z_int,"quadratic")

        for i in range(s,e):
                z[i] = z_new[i-s]

        for p in range(1, le - 1):
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

        

        fig_p, axs_p = plt.subplots(1, 3, figsize=(10, 3), subplot_kw=dict(projection="3d"))
        le = len(rot_x_store)
        p_list = []
        for p in range(le-1):
                # get proceeding circle (as x,y,z samples)
                x, y, z = create_circle_known(
                        [data[keys[i]][p + 1] for i in range(len(keys))],rot_x_store[p],rot_z_store[p]
                )
                p_list.append([x,y,z])
        p_list = np.array(p_list)

        for i in range(len(p_list[:,:,:])-1):
                # plot for reference
                col = 'k'
                for ax in axs_p:
                        ax.plot3D(p_list[i,0,:],p_list[i,1,:],p_list[i,2,:], c=col, lw=0.5,alpha=0.25)
                        for j in np.linspace(0,len(p_list[i,0,:]),10,endpoint=False):
                                j = int(j)
                                ax.plot3D([p_list[i,0,j],p_list[i+1,0,j]],[p_list[i,1,j],p_list[i+1,1,j]],[p_list[i,2,j],p_list[i+1,2,j]],c=col,lw=0.5,alpha=0.25)


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




        # plt.subplots_adjust(left=0.01,right=0.99,wspace=0,top=1)
        # plt.show()
        # copy existing base mesh folder
        
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

        axs_p[0].set_xlabel("x",fontsize=14)
        axs_p[0].set_zlabel("z",fontsize=14)

        axs_p[1].set_ylabel("y",fontsize=14)
        axs_p[1].set_zlabel("z",fontsize=14)

        axs_p[2].set_ylabel("y",fontsize=14)
        axs_p[2].set_xlabel("x",fontsize=14)

        axs_p[0].view_init(0, 270)
        axs_p[1].view_init(0, 180)
        axs_p[2].view_init(270, 0)


        fig_p.tight_layout()
        fig_p.savefig(path + "/pre-render.png", dpi=600)
        # fig.savefig(path + "/interp.pdf", dpi=200)

        # # run script to create mesh
        print("Writing geometry")
        mesh.write(output_path=os.path.join(path, "system", "blockMeshDict"), geometry=None)
        print("Running blockMesh")
        os.system("chmod +x " + path + "/Allrun.mesh")
        os.system(path + "/Allrun.mesh")

        return


coils = 2  # number of coils
h = coils * 0.010391  # max height
N = 2 * np.pi * coils  # angular turns (radians)
n = 7  # points to use

nominal_data = {}

z_vals = np.linspace(0, h/2, n)
theta_vals = np.flip(np.linspace(0+np.pi/2, N+np.pi/2, n))
rho_vals = [0.0125/2 for i in range(n)]
tube_rad_vals = [0.0025/2 for i in range(n)]
for i in range(n):
	nominal_data["z_" + str(i)] = z_vals[i]
	nominal_data["theta_" + str(i)] = theta_vals[i]
	nominal_data["tube_rad_" + str(i)] = tube_rad_vals[i]
	nominal_data["rho_" + str(i)] = rho_vals[i]

data = {}
data['fid_radial'] = 4
data['fid_axial'] = 40

data['rho_0'] = 0
data['z_0'] = 0
data['z_1'] = np.random.uniform(-0.0015,0.0015)
data['rho_1'] = 0
for i in range(2,n):
        data['z_'+str(i)] = np.random.uniform(-0.0015,0.0015)
        data['rho_'+str(i)] = np.random.uniform(-0.0035,0.0035)

create_mesh(data,'mesh_generation/test',n,nominal_data)