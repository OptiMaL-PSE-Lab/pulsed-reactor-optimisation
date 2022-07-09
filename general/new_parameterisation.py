from re import A
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
from classy_examples.classy_blocks.classes.primitives import Edge
from classy_examples.classy_blocks.classes.block import Block
from classy_examples.classy_blocks.classes.mesh import Mesh




def rotate_z(x,y,z,r_z):
	x = np.array(x)
	y = np.array(y)
	z = np.array(z)
	x_new = x*np.cos(r_z)-y*np.sin(r_z)
	y_new = x*np.sin(r_z)+y*np.cos(r_z)
	return x_new,y_new,z

def create_ellipse(c_x,c_y,r,t,c_z):
	alpha = np.linspace(0, 2*np.pi, 100)
	z = r*np.cos(alpha)+c_z
	x = r*np.sin(alpha)+c_x
	y = [c_y for i in range(len(z))]
	x-=c_x
	y-=c_y 
	x,y,z = rotate_z(x,y,z,t)
	x+=c_x
	y+=c_y
	return x,y,z

def interpolate(y,f,kind):
	x = np.linspace(0, len(y),len(y))
	x_new = np.linspace(0,len(y),len(y)*f)
	f = interp1d(x, y,kind=kind)
	y_new = f(x_new)
	# plt.figure()
	# plt.scatter(x, y)
	# plt.plot(x_new,y_new)
	# plt.show()
	return y_new


n = 20
coil_rad = 2
rot = 2
factor = 10
radius = 0.5

x0 = 0
y0 = 0
z0 = 0 
t0 = 0 
r0 = 0.5

x_plot = np.diff([(coil_rad*np.cos(x_y)) for x_y  in np.linspace(0,2*rot*np.pi,n)])
y_plot = np.diff([(coil_rad*np.sin(x_y)) for x_y  in np.linspace(0,2*rot*np.pi,n)])
t_plot = np.diff(np.linspace(0,2*rot*np.pi,n))
r_plot = np.array([0 for i in range(n)])
z_plot = np.diff(np.linspace(-3,3,n))


x_plot = np.cumsum(np.append([x0],x_plot))
y_plot = np.cumsum(np.append([y0],y_plot))
t_plot = np.cumsum(np.append([t0],t_plot))
r_plot = np.cumsum(np.append([r0],r_plot))
z_plot = np.cumsum(np.append([z0],z_plot))

x_plot = interpolate(x_plot,factor,'quadratic')
y_plot = interpolate(y_plot,factor,'quadratic')
t_plot = interpolate(t_plot,factor,'quadratic')
r_plot = interpolate(r_plot,factor,'quadratic')
z_plot = interpolate(z_plot,factor,'quadratic')



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
i = 0 
x,y,z = create_ellipse(x_plot[i],y_plot[i],r_plot[i],t_plot[i],z_plot[i])
ax.plot3D(x,y,z,c='k')


i += 1 
x_next,y_next,z_next = create_ellipse(x_plot[i],y_plot[i],r_plot[i],t_plot[i],z_plot[i])
ax.plot3D(x_next,y_next,z_next,c='k')


for i in range(1,len(x_plot)):
	x_next,y_next,z_next = create_ellipse(x_plot[i],y_plot[i],r_plot[i],t_plot[i],z_plot[i])
	ax.plot3D(x_next,y_next,z_next,c='k')

plt.show()

mesh = Mesh()

block_points = [
[0, 0, 0],
[1, 0, 0],
[1, 1, 0],
[0, 1, 0],

[0, 0, 1],
[1, 0, 1],
[1, 1, 1],
[0, 1, 1],
]

block_edges = [
Edge(0, 1, [0.5, -0.25, 0]), # arc edges
Edge(4, 5, [0.5, -0.1, 1]),

Edge(2, 3, [[0.7, 1.3, 0], [0.3, 1.3, 0]]), # spline edges
Edge(6, 7, [[0.7, 1.1, 1], [0.3, 1.1, 1]])
]

# the most low-level way of creating a block is from 'raw' points
block = Block.create_from_points(block_points, block_edges)
block.set_patch(['left', 'right', 'front', 'back'], 'walls')
block.set_patch('bottom', 'inlet')

block.chop(0, start_size=0.02, c2c_expansion=1.1)
block.chop(1, start_size=0.01, c2c_expansion=1.2)
block.chop(2, start_size=0.1, c2c_expansion=1)

mesh.add_block(block)

# another block!
block_points = block_points[4:] + [
[0, 0, 1.7],
[1, 0, 1.8],
[1, 1, 1.9],
[0, 1, 2],
]
block = Block.create_from_points(block_points)
block.set_patch(['left', 'right', 'front', 'back'], 'walls')
block.set_patch('top', 'outlet')

block.chop(2, length_ratio=0.5, start_size=0.02, c2c_expansion=1.2, invert=False)
block.chop(2, length_ratio=0.5, start_size=0.02, c2c_expansion=1.2, invert=True)

mesh.add_block(block)




