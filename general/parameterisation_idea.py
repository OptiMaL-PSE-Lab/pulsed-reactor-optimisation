import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d


fig = plt.figure()
ax = plt.axes(projection='3d')
def add_circle(x,y,z,r,ax):
	p = Circle((x,y), r,color='k',alpha=0.1,edgecolor='k')
	ax.add_patch(p)
	art3d.pathpatch_2d_to_3d(p, z=z, zdir="z")
	p_edge = Circle((x,y), r,edgecolor='k',fill=False,alpha=0.5)
	ax.add_patch(p_edge)
	art3d.pathpatch_2d_to_3d(p_edge, z=z, zdir="z")
	p_1 = np.array([x-r,y,z])
	p_2 = np.array([x+r,y,z])
	return p_1,p_2

n = 100
z_lim = 10
z_vals = np.linspace(0,z_lim,n)
r_vals = np.ones(n)*0.3
x_vals = np.sin(np.linspace(0,2.5*np.pi*2,n))
y_vals = np.cos(np.linspace(0,2.5*np.pi*2,n))

p1_store = np.array([[0,0,0]])
p2_store = np.array([[0,0,0]])

for i in range(n):
	p1,p2 = add_circle(x_vals[i],y_vals[i],z_vals[i],r_vals[i],ax)
	p1_store = np.append(p1_store,[p1],axis=0)
	p2_store = np.append(p2_store,[p2],axis=0)

ax.plot3D(p1_store[1:,0],p1_store[1:,1],p1_store[1:,2],c='k',alpha=0.5)
ax.plot3D(p2_store[1:,0],p2_store[1:,1],p2_store[1:,2],c='k',alpha=0.5)

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(0, z_lim)

plt.show()