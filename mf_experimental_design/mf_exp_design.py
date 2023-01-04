import sys
import os
import pickle 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from mf_experimental_design.utils import *
from mesh_generation.coil_basic import create_mesh
import time
import gpjax as gpx 
from jax import grad, jit
import jax.numpy as jnp
from uuid import uuid4
import jax.random as jr
import optax as ox
import json 



def eval_cfd(x: dict):
    start = time.time()
    #ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ID = str(uuid4())
    print("Starting to mesh " + ID)
    case = "outputs/mf/" + ID
    create_mesh(
        x,
        length=0.0753,
        tube_rad =0.0025,
        path=case,
    )
    parse_conditions(case, x)
    times, values = run_cfd(case)
    N = calculate_N(values, times, case)
    for i in range(16):
        shutil.rmtree(case + "/processor" + str(i))
    # shutil.rmtree(newcase)
    end = time.time()
    return {'obj':N,'cost':end-start,'id':ID}


bounds = {}
bounds['a'] = [0.001,0.008]
bounds['f'] = [2,8]
bounds['re'] = [10,50]
bounds['pitch'] = [0.0075,0.015]
bounds['coil_rad'] = [0.003,0.0125]
bounds['inversion_loc'] = [0,1]
bounds['fid_axial'] = [0,1]
bounds['fid_radial'] = [0,1]

def sample_bounds(bounds,n):
        sample = lhs(np.array(list(bounds.values())),n)
        return sample

def sample_to_dict(sample):
        sample_dict = {}
        keys = list(bounds.keys())
        for i in range(len(sample)):
                sample_dict[keys[i]] = sample[i]
        return sample_dict


n = 25
samples = sample_bounds(bounds,n)

data = {'data':[]}
for sample in samples:
        sample_dict = sample_to_dict(sample)
        print(sample_dict)
        res = eval_cfd(sample_dict)
        data['data'].append({'id':res['id'],'x':sample_dict,'cost':res['cost'],'obj':res['obj']})
        print(data)
        with open('outputs/data.json', 'w') as fp:
                json.dump(data, fp)


# def train_gp(inputs,outputs):
#         D = gpx.Dataset(X=inputs, y=outputs)
#         kern = gpx.RBF(active_dims=[i for i in range(D.in_dim)])
#         prior = gpx.Prior(kernel = kern)
#         likelihood = gpx.Gaussian(num_datapoints = D.n)
#         posterior = prior * likelihood

#         mll = jit(posterior.marginal_log_likelihood(D, negative=True))
#         opt = ox.adam(learning_rate=1e-3)
#         parameter_state = gpx.initialise(posterior)
#         inference_state = gpx.fit(mll, parameter_state, opt, n_iters=500)

#         learned_params, _ = inference_state.unpack()
#         return posterior, learned_params, D, likelihood

# def inference(posterior,learned_params,D,likelihood,inputs):
#         latent_distribution = posterior(learned_params, D)(inputs)
#         predictive_distribution = likelihood(learned_params, latent_distribution)
#         predictive_mean = predictive_distribution.mean()
#         predictive_cov = predictive_distribution.covariance()
#         if len(inputs) == 1:
#                 return predictive_mean.item(),predictive_cov.item()
#         else:
#                 return predictive_mean,predictive_cov

# posterior,learned_params,D,likelihood = train_gp(sample,outputs)

# xtest = sample_bounds(bounds,1)

# print(inference(posterior,learned_params,D,likelihood,xtest))


# # res = eval_cfd(sample_to_dict(sample[0,:]))

