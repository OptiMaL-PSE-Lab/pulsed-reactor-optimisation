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

low_fid = read_json('symbolic_mf_data_generation/low/data.json')['data']
high_fid = read_json('symbolic_mf_data_generation/high/data.json')['data']
dt = 0.0025
low_in = [d['x'] for d in low_fid]
low_out = np.array([d['obj'] for d in low_fid])
high_in = [d['x'] for d in high_fid]
high_out = np.array([d['obj'] for d in high_fid])


def add_dimensionless_numbers(x):
    st = x['coil_rad']/(2*np.pi*x['a'])
    re0 = (2*np.pi*x['f']*x['a']*990*dt)/(9.9*10**(-4))
    de = re0 / (np.sqrt(dt/(2*x['coil_rad'])))
    x['st'] = st
    x['de'] = de
    return x 

for l in low_in:
    l = add_dimensionless_numbers(l)
for h in high_in:
    h = add_dimensionless_numbers(h)




keys = ['st','de','re']
low_in = np.array([[l[k] for k in keys] for l in low_in])
high_in = np.array([[h[k] for k in keys] for h in high_in])
keys = ['st','de','rey']

file_name = 'symbolic_mf_data_generation/low_fid.csv'

low_model = PySRRegressor(
	niterations=10000,
	binary_operators=["+", "*", "/", "-","pow"],
	unary_operators=["exp","log","neg","square","cube"],
	model_selection='score',
	maxsize = 30,
	maxdepth=10,
	timeout_in_seconds = 240,
	parsimony = 1e-5,
	equation_file= file_name
)
low_model.fit(low_in,low_out.reshape(-1,1),variable_names = keys)

print(low_model.latex())
def low_fid(x):
	if x.__class__ != float or x.__class__ != int:
		return low_model.sympy()
	else:
		return low_model.sympy().evalf(x)
