import numpy as np 
import matplotlib.pyplot as plt 

def create_ellipse(c_x,c_y,r_x,r_y,theta):
	alpha = np.linspace(0, 2*np.pi, 100)
	x_a = r_x*np.cos(alpha)*np.cos(theta) - r_y*np.sin(alpha)*np.sin(theta)+c_x
	y_a = r_x*np.cos(alpha)*np.sin(theta) + r_y*np.sin(alpha)*np.cos(theta)+c_y
	return x_a,y_a 

def add_ellipse(p,ax):
	x = np.cumsum([p['x0']]+p['dx'])
	y = np.cumsum([p['y0']]+p['dy'])
	rx = np.cumsum([p['rx0']]+p['drx'])
	ry = np.cumsum([p['ry0']]+p['dry'])
	print(ry)
	z = np.cumsum([p['z0']]+p['dz'])
	t = np.cumsum([p['t0']]+p['dt'])
	for i in range(len(x)):
		x_a,y_a = create_ellipse(x[i],y[i],rx[i],ry[i],t[i])
		ax.plot3D(x_a,y_a,np.array([z[i] for j in range(len(x_a))]),color='k',alpha=0.5)
	return 





n = 50
p = {}
# PARAMETERS
p['z0'] = 0 
p['x0'] = 0 
p['y0'] = 0 
p['rx0'] = 1
p['ry0'] = 1
p['t0'] = 0 
p['dz']  = [10/n for i in range(n)]

# VARIABLES (BOUNDS ENSURE CONTINUITY)
p['drx'] = list(np.random.uniform(-0.2,0.1,n)) 
p['dry'] = list(np.random.uniform(-0.2,0.1,n))
p['dt']  = list(np.random.uniform(-0.3,0.3,n)) 
p['dx']  =  np.diff(np.sin(np.linspace(0,2.5*np.pi*2,n))) # -0.3 <--> 0.3
p['dy'] = np.diff(np.cos(np.linspace(0,2.5*np.pi*2,n)))  # -0.3 <--> 0.3


# 1. KEEP DX AND DY A HELIX 
# 2. DON'T... BUT IMPOSE SMOOTH CONDITIONS
# 3. CONSTRAIN START AND END TO BE A 'CIRCLE'.... IN SAME PLACE?


#TODO 
# PLOT SAMPLES 
# DEFINE CONSTRAINTS 
# DEFINE VARIANTS




fig = plt.figure()
ax = plt.axes(projection='3d')
add_ellipse(p,ax=ax)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(0, 10)

plt.show()