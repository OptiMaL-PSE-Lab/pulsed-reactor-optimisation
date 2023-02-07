import sys
import os
import pickle
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from mf_experimental_design.utils import *
from mf_experimental_design.mf_exp_design import * 
from mesh_generation.coil_basic import create_mesh
import time
import gpjax as gpx
from jax import grad, jit, value_and_grad
import jax.numpy as jnp
from uuid import uuid4
import jax.random as jr
import optax as ox
import jax
from scipy.optimize import minimize
import json


n = 32

x_bounds = {}
x_bounds["a"] = [0.001, 0.008]
x_bounds["f"] = [2, 8]
x_bounds["re"] = [10, 50]
x_bounds["pitch"] = [0.0075, 0.02]
x_bounds["coil_rad"] = [0.0035, 0.0125]
x_bounds["inversion_loc"] = [0, 1]

z_bounds = {}
z_bounds["fid_axial"] = [20.001, 49.99]
z_bounds["fid_radial"] = [1.001, 5.99]
n_fid = len(z_bounds)

joint_bounds = x_bounds | z_bounds

x_bounds_og = x_bounds.copy()
z_bounds_og = z_bounds.copy()
joint_bounds_og = joint_bounds.copy()


samples = sample_bounds(joint_bounds,n)
data_path = 'static_data_generation/data.json'
data = {'data':[]}
for sample in samples:
        for i in range(n_fid):
                sample[-(i+1)] = int(sample[-(i+1)])
        sample_dict = sample_to_dict(sample,joint_bounds)
        run_info = {'id':'running','x':sample_dict,'cost':'running','obj':'running'}
        data['data'].append(run_info)
        save_json(data,data_path)
        res = eval_cfd(sample_dict)
        run_info = {'id':res['id'],'x':sample_dict,'cost':res['cost'],'obj':res['obj']}
        data['data'][-1] = run_info
        save_json(data,data_path)


