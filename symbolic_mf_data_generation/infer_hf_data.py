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
from tqdm import tqdm 


path = 'symbolic_mf_data_generation/exp_design/data.json'
data = read_json(path)['data'][:80]

inputs = np.array([list(d['x'].values()) for d in data])
keys = list(data[0]['x'].keys())

outputs = np.array([d['obj'] for d in data]).reshape(-1,1)

m_inputs,std_inputs = mean_std(inputs)
m_outputs,std_outputs = mean_std(outputs)

outputs = normalise(outputs,m_outputs,std_outputs)
inputs = normalise(inputs,m_inputs,std_inputs)

gp = build_gp_dict(*train_gp(inputs,outputs,ms=1))

n_sample = 5000
bounds = np.array([[np.min(inputs[:,i]),np.max(inputs[:,i])] for i in range(inputs.shape[1])])

x_sample = lhs(bounds, n_sample,log=False)
new_data = []
for i in range(len(x_sample)):
	x_sample[i,6] = np.max(x_sample[:,6])
	x_sample[i,7] = np.max(x_sample[:,7])
	# create dict with keys
	new_x = unnormalise(x_sample[i],m_inputs,std_inputs)
	new_x = dict(zip(keys,new_x))
	new_data.append({'x':new_x})

new_data = {'data':new_data}

mean = []
cov = []
for x in tqdm(x_sample):
	mean_v, cov_v = inference(gp, jnp.array([x]))
	mean.append(mean_v)
	cov.append(cov_v)


for i in range(len(x_sample)):
	new_data['data'][i]['mean'] = float(unnormalise(float(mean[i]),m_outputs,std_outputs))
	new_data['data'][i]['std'] = float(unnormalise(float(np.sqrt(cov[i])),m_outputs,std_outputs))
	new_data['data'][i]['obj'] = float(np.random.normal(new_data['data'][i]['mean'],new_data['data'][i]['std']))

save_json(new_data,'symbolic_mf_data_generation/exp_design/interp_data_dist.json')

new_data = {'data':new_data}
for i in range(len(x_sample)):
	new_data['data'][i]['mean'] = float(unnormalise(float(mean[i]),m_outputs,std_outputs))
	new_data['data'][i]['std'] = float(unnormalise(float(np.sqrt(cov[i])),m_outputs,std_outputs))
	new_data['data'][i]['obj'] =  float(unnormalise(float(mean[i]),m_outputs,std_outputs))

save_json(new_data,'symbolic_mf_data_generation/exp_design/interp_data_mean.json')