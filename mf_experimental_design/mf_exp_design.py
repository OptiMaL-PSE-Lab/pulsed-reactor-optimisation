import sys
import os
import pickle 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from mf_experimental_design.utils import *
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



def eval_cfd(x: dict):
    start = time.time()
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
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



def sample_bounds(bounds,n):
        sample = lhs(jnp.array(list(bounds.values())),n)
        return sample


def sample_to_dict(sample,bounds):
        sample_dict = {}
        keys = list(bounds.keys())
        for i in range(len(sample)):
                sample_dict[keys[i]] = float(sample[i])
        return sample_dict


def read_json(path):
        with open(path, 'r') as f:
                data = json.load(f)
        return data

def save_json(data,path):
        with open(path, 'w') as f:
                json.dump(data,f)
        return 


def format_data(data):
        inputs = []
        obj = []
        cost = []
        for d in data['data']:
                inputs += [list(d['x'].values())]
                obj += [d['obj']]
                cost += [d['cost']]

        inputs = jnp.array(inputs)
        obj = jnp.array(obj).reshape(-1,1)
        cost = jnp.array(cost).reshape(-1,1)
        return inputs,obj,cost


def train_gp(inputs,outputs):
        D = gpx.Dataset(X=inputs, y=outputs)
        kern = gpx.RBF(active_dims=[i for i in range(D.in_dim)])
        prior = gpx.Prior(kernel = kern)
        likelihood = gpx.Gaussian(num_datapoints = D.n)
        posterior = prior * likelihood

        mll = jit(posterior.marginal_log_likelihood(D, negative=True))
        opt = ox.adam(learning_rate=1e-2)
        parameter_state = gpx.initialise(posterior)
        inference_state = gpx.fit(mll, parameter_state, opt, num_iters=50000)

        learned_params, _ = inference_state.unpack()
        return posterior, learned_params, D, likelihood

def inference(gp,inputs):
        posterior = gp['posterior']
        learned_params = gp['learned_params']
        D = gp['D']
        likelihood = gp['likelihood']
        latent_distribution = posterior(learned_params, D)(inputs)
        predictive_distribution = likelihood(learned_params, latent_distribution)
        predictive_mean = predictive_distribution.mean()
        predictive_cov = predictive_distribution.covariance()
        return predictive_mean,predictive_cov



def mean_std(vec):
        return np.mean(vec,axis=0),np.std(vec,axis=0)

def normalise(vec,mean,std):
        return (vec-mean)/std

def unnormalise(vec,mean,std):
        return (vec*std)+mean

def normalise_bounds_dict(bounds,mean,std):
        keys = list(bounds.keys())
        new_bounds = {}
        for i in range(len(keys)):
                original_bounds = np.array(bounds[keys[i]])
                normalised_bounds = (original_bounds-mean[i])/std[i]
                new_bounds[keys[i]] = list(normalised_bounds)
        return new_bounds

def build_gp_dict(posterior,learned_params,D,likelihood):
        gp_dict = {}
        gp_dict['posterior'] = posterior
        gp_dict['learned_params'] = learned_params 
        gp_dict['D'] = D
        gp_dict['likelihood'] = likelihood
        return gp_dict

def aquisition_function(x,gp,cost_gp,fid_high,gamma,beta):
        cost,cost_var = inference(cost_gp,jnp.array([x]))
        # fixing fidelity
        for i in range(n_fid):
                i += 1 
                x = x.at[-i].set(fid_high[-i])
        mean,cov = inference(gp,jnp.array([x]))
        return -((mean[0]+beta*cov[0])/(gamma*cost[0]))[0]


n = 25

x_bounds = {}
x_bounds['a'] = [0.001,0.008]
x_bounds['f'] = [2,8]
x_bounds['re'] = [10,50]
x_bounds['pitch'] = [0.0075,0.02]
x_bounds['coil_rad'] = [0.0035,0.0125]
x_bounds['inversion_loc'] = [0,1]

z_bounds = {}
z_bounds['fid_axial'] = [20.001,49.99]
z_bounds['fid_radial'] = [1.001,5.99]
n_fid = len(z_bounds)

joint_bounds = x_bounds | z_bounds 

x_bounds_og = x_bounds.copy()
z_bounds_og = z_bounds.copy()
joint_bounds_og = joint_bounds.copy()

# samples = sample_bounds(joint_bounds,n)
# data_path = 'outputs/mf/data.json'
# data = {'data':[]}
# for sample in samples:
#         for i in range(n_fid):
#                 sample[-(i+1)] = int(sample[-(i+1)])
#         sample_dict = sample_to_dict(sample,joint_bounds)
#         run_info = {'id':'running','x':sample_dict,'cost':'running','obj':'running'}
#         data['data'].append(run_info)
#         save_json(data,data_path)
#         res = eval_cfd(sample_dict)
#         run_info = {'id':res['id'],'x':sample_dict,'cost':res['cost'],'obj':res['obj']}
#         data['data'][-1] = run_info
#         save_json(data,data_path)

data_path = 'outputs/mf/data.json'

while True:
        # reading data from file format
        data = read_json(data_path)
        inputs,outputs,cost = format_data(data)

        # normalising all data

        j_mean,j_std = mean_std(inputs)
        inputs = normalise(inputs,j_mean,j_std)
        x_mean = j_mean[:-n_fid]
        x_std = j_std[:-n_fid]
        z_mean = j_mean[-n_fid:]
        z_std = j_std[-n_fid:]

        o_mean,o_std = mean_std(outputs)
        outputs = normalise(outputs,o_mean,o_std)
        c_mean,c_std = mean_std(cost)
        cost = normalise(cost,c_mean,c_std)

        x_bounds = normalise_bounds_dict(x_bounds_og,x_mean,x_std)
        z_bounds = normalise_bounds_dict(z_bounds_og,z_mean,z_std)
        joint_bounds = normalise_bounds_dict(joint_bounds_og,j_mean,j_std)


        # training two Gaussian processes:
        print('Training GPs')
        # all inputs and fidelities against objective
        gp = build_gp_dict(*train_gp(inputs,outputs)) 
        # inputs and fidelities against cost
        cost_gp = build_gp_dict(*train_gp(inputs,cost)) 

        # optimising the aquisition of inputs, disregarding fidelity
        print('optimising aquisition function')

        gamma = 0.5
        beta = 2.5 

        def optimise_aquisition(cost_gp,gp,ms_num,gamma,beta):
                # normalise bounds
                b_list = list(joint_bounds.values())
                fid_high = jnp.array([list(z_bounds.values())[i][1] for i in range(n_fid)])
                # sample and normalise initial guesses
                x_init = jnp.array(sample_bounds(joint_bounds,ms_num))
                f_best = 1E20
                f = value_and_grad(aquisition_function)
                run_store = []
                for i in range(ms_num):
                        x = x_init[i]
                        res = minimize(f,x0=x,args=(gp,cost_gp,fid_high,gamma,beta),method='SLSQP',bounds=b_list,jac=True,options={'disp':False})
                        aq_val = res.fun
                        x = res.x
                        run_store.append(aq_val)
                        if aq_val < f_best:
                                f_best = aq_val
                                x_best = x
                return x_best,aq_val

        x_opt,f_opt = optimise_aquisition(cost_gp,gp,16,gamma,beta)
        print('normalised res:',x_opt)
        x_opt = list(unnormalise(x_opt,j_mean,j_std))
        print('unnormalised res:',x_opt)

        for i in range(n_fid):
                x_opt[-(i+1)] = int(x_opt[-(i+1)])

        sample = sample_to_dict(x_opt,joint_bounds)

        print('Running ',sample)
        run_info = {'id':'running','x':sample,'cost':'running','obj':'running'}
        data['data'].append(run_info)
        save_json(data,data_path)

        res = eval_cfd(sample)
        run_info = {'id':res['id'],'x':sample,'cost':res['cost'],'obj':res['obj']}
        data['data'][-1] = run_info
        save_json(data,data_path)

