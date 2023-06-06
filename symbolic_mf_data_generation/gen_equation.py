import sys 
import os 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import jax.numpy as jnp
from utils import *
from utils_plotting import *
from utils_gp import *
import numpy as np
import jax
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from pysr import PySRRegressor
from julia.api import Julia
os.system('''export PATH="$PATH:julia-1.8.5/bin"''')



def gen_equation(path,inputs,outputs,keys):
	
	data_name = path.split('/')[-1]
	file_name = path.split(data_name)[0] + '/equation.csv'

	model = PySRRegressor(
		niterations=10000,
		binary_operators=["+", "*", "/", "-","^"],
		unary_operators=["exp","log","neg","square","cube","sin"],
		model_selection='best',
		maxsize = 40, 
		turbo=True,
		loss="L1DistLoss()",
		maxdepth = 20,
		timeout_in_seconds = 360,
		verbosity=0,
		parsimony = 1e-5,
		equation_file= file_name,
		progress=False
	)

	model.fit(inputs,outputs.reshape(-1,1),variable_names = keys)

	print(model.latex())
	def f(x):
		if x.__class__ != float or x.__class__ != int:
			return model.sympy()
		else:
			return model.sympy().evalf(x)
		
	return 


def add_dimensionless_numbers(x):
	dt = 0.0025
	st = x['coil_rad']/(2*np.pi*x['a'])
	re0 = (2*np.pi*x['f']*x['a']*990*dt)/(9.9*10**(-4))
	de = re0 / (np.sqrt(dt/(2*x['coil_rad'])))
	x['st'] = st
	x['de'] = de
	x['rey'] = x['re']
	return x 

# path = 'symbolic_mf_data_generation/exp_design/interp_data_dist.json'
# data = read_json(path)['data']
# inputs = [d['x'] for d in data]
# outputs = np.array([d['mean'] for d in data])
# keys = ['st','de','rey','coil_rad','pitch','a','f']
# inputs = np.array([[l[k] for k in keys] for l in inputs])
# gen_equation(path,inputs,outputs,keys)



path = 'symbolic_mf_data_generation/toy/hf_data.json'
data = read_json(path)['data']
inputs = [d['x'] for d in data]
outputs = np.array([d['obj'] for d in data])
keys = ['x1']
inputs = np.array([[l[k] for k in keys] for l in inputs])
gen_equation(path,inputs,outputs,keys)



