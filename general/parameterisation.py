from configparser import Interpolation
from ctypes import c_byte
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d


def create_ellipse(c_x,c_y,r_x,r_y,theta,warp):
	alpha = np.linspace(0, 2*np.pi, 100)
	x_a = r_x*np.cos(alpha)+c_x
	y_a = r_y*np.sin(alpha)+c_y
	x_test = np.linspace(0,np.pi,50)
	y_test = np.abs(np.sin(x_test))
	y_test_use = np.append(y_test,y_test)
	y_a_warp = y_a + [y_test_use[i]*abs(y_a[i])*warp for i in range(len(y_test_use))]

	x_a_r = r_x*np.cos(alpha)*np.cos(theta) - r_y*np.sin(alpha)*np.sin(theta)+c_x
	y_a_r = r_x*np.cos(alpha)*np.sin(theta) + r_y*np.sin(alpha)*np.cos(theta)+c_y
	return x_a,y_a,x_a_r,y_a_r,y_test,x_test,y_a_warp

plt.figure()
c_x = 0
c_y = 0
r_x = 1
r_y = 0.5 
t = 2
warp=1
x,y,x_r,y_r,y_test,x_test,y_w = create_ellipse(c_x,c_y,r_x,r_y,t,warp)


#plt.plot(x_r,y_r,c='k',ls='dashed')
plt.plot(x,y,c='k')
plt.plot(x,y_w,c='r',ls='dashed')
plt.savefig('test.png')



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



def add_ellipse(p,ax,**kwargs):
	x = np.cumsum([p['x0']]+p['dx'])
	y = np.cumsum([p['y0']]+p['dy'])
	rx = np.cumsum([p['rx0']]+p['drx'])
	ry = np.cumsum([p['ry0']]+p['dry'])
	z = np.cumsum([p['z0']]+p['dz'])
	t = np.cumsum([p['t0']]+p['dt'])
	# first check if kwargs exist
	try:
		interpolation = kwargs['interpolation']
		f = kwargs['extension']
	# if not don't do interpolation
	except KeyError:
		# plot each ellipse
		for i in range(len(x)):
			x_a,y_a = create_ellipse(x[i],y[i],rx[i],ry[i],t[i])
			ax.plot3D(x_a,y_a,np.array([z[i] for j in range(len(x_a))]),color='k',alpha=0.5)
		return 
	

	x = interpolate(x,f,interpolation)
	y = interpolate(y,f,interpolation)
	rx = interpolate(rx,f,interpolation)
	ry = interpolate(ry,f,interpolation)
	z = interpolate(z,f,interpolation)
	t = interpolate(t,f,interpolation)
	

	for i in range(len(x)):
		x_a,y_a = create_ellipse(x[i],y[i],rx[i],ry[i],t[i])
		ax.plot3D(x_a,y_a,np.array([z[i] for j in range(len(x_a))]),color='k',alpha=0.5)
	return 


# n = 50
# p = {}
# # PARAMETERS
# p['z0'] = 0 
# p['x0'] = 0 
# p['y0'] = 0 
# p['rx0'] = 0.25
# p['ry0'] = 0.5
# p['t0'] = 0 
# p['dz']  = [10/n for i in range(n)]


# # VARIABLES (BOUNDS ENSURE CONTINUITY)
# p['drx'] = list(np.random.uniform(-0.0,0.0,n)) 
# p['dry'] = list(np.random.uniform(-0.0,0.0,n))
# coils = 3
# p['dx']  =  np.diff(np.cos(np.linspace(0,coils*2*np.pi,n))) # -0.3 <--> 0.3
# p['dy'] = np.diff(np.sin(np.linspace(0,coils*2*np.pi,n)))  # -0.3 <--> 0.3
# p['dt']  = np.diff(np.linspace(0,2*coils*np.pi,n))


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# add_ellipse(p,ax=ax,interpolation='quadratic',extension=2)
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)
# ax.set_zlim(0, 10)

# plt.show()


# #TODO 
# # SIMPLIFY CODE




