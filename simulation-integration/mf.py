from bayes_opt_with_constraints.bayes_opt import BayesianOptimization, UtilityFunction
from datetime import datetime
from utils import newJSONLogger, vel_calc, parse_conditions, run_cfd, calculate_N
from bayes_opt_with_constraints.bayes_opt.event import Events
import json
import os 
import pickle 
import numpy as np 
from uuid import uuid4
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from mesh_generation.coil_basic import create_mesh
import shutil 
from re import S
import numpy as np 
import matplotlib.pyplot as plt 
import GPy 
import numpy as np 
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
import matplotlib.pyplot as plt 
import numpy as np
from emukit.examples.multi_fidelity_dgp.multi_fidelity_deep_gp import DGP_Base, init_layers_mf
from gpflow.kernels import RBF, White, Linear
from gpflow.likelihoods import Gaussian
from gpflow.actions import Loop, Action
from gpflow.mean_functions import Zero
from gpflow.training import AdamOptimizer
import gpflow.training.monitor as mon
from gpflow.training import AdamOptimizer, ScipyOptimizer, NatGradOptimizer

def make_dgpMF_model(X, Y, Z):
    
    L = len(X)

    Din = X[0].shape[1]
    Dout = Y[0].shape[1]

    kernels = []
    k_2 = RBF(Din, active_dims=list(range(Din)), variance=1., lengthscales=10., ARD=True)
    kernels.append(k_2)
    for l in range(1,L):
        
        D = Din + Dout
        D_range = list(range(D))
        k_corr_2 = RBF(Din, active_dims=D_range[:Din], lengthscales=0.1,  variance=1.5, ARD=True)
        k_corr = k_corr_2
        
        k_prev = RBF(Dout, active_dims=D_range[Din:], variance = 1., lengthscales=0.1, ARD=True)
        k_in = RBF(Din, active_dims=D_range[:Din], variance=0.1, lengthscales=1., ARD=True)
        k_bias = Linear(Dout, active_dims=D_range[Din:], variance = 1e-6)
        k_in.variance = 1e-6
        k_l = k_corr*(k_prev + k_bias) + k_in
        kernels.append(k_l)

    '''
    A White noise kernel is currently expected by Mf-DGP at all layers except the last.
    In cases where no noise is desired, this should be set to 0 and fixed, as follows:
    
        white = White(1, variance=0.)
        white.variance.trainable = False
        kernels[i] += white
    '''
    for i, kernel in enumerate(kernels[:-1]):
        kernels[i] += White(1, variance=0.)
            
    num_data = 0
    for i in range(len(X)):
        print('\nData at Fidelity ', (i+1))
        print('X - ', X[i].shape)
        print('Y - ', Y[i].shape)
        print('Z - ', Z[i].shape)
        num_data += X[i].shape[0]
        
    layers = init_layers_mf(Y, Z, kernels, num_outputs=1)
        
    model = DGP_Base(X, Y, Gaussian(), layers, num_samples=10, minibatch_size=1000)

    return model

class PrintAction(Action):
    def __init__(self, model, text):
        self.model = model
        self.text = text
        
    def run(self, ctx):
        if ctx.iteration % 10 == 0:
            likelihood = ctx.session.run(self.model.likelihood_tensor)
            objective = ctx.session.run(self.model.objective)

            print('ELBO {:.4f};  KL {:,.4f}'.format(ctx.session.run(self.model.L), ctx.session.run(self.model.KL)))
            print('{}: iteration {} objective {:,.4f}'.format(self.text, ctx.iteration, objective))

def run_adam(model, lr, iterations, callback=None):
    adam = AdamOptimizer(lr).make_optimize_action(model)
    callback = PrintAction(model,'')
    actions = [adam] if callback is None else [adam, callback]
    loop = Loop(actions, stop=iterations)()
    model.anchor(model.enquire_session())

def eval_cfd(a, f, re, coil_rad, pitch, inversion_loc):
    fid = [1,1]
    tube_rad = 0.0025
    length = 0.0753
    identifier = identifier = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print('Starting to mesh '+identifier)
    newcase = "outputs/geom/" + identifier
    create_mesh(coil_rad, tube_rad, pitch, length, inversion_loc, fid,path=newcase,validation=True,build=True)
    vel = vel_calc(re)
    parse_conditions(newcase, a, f, vel)
    time, value = run_cfd(newcase)
    N = calculate_N(value, time,newcase)
    #shutil.rmtree(newcase)
    return N

def parse_initial_conditions(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
    X = np.zeros((len(data[0]['x']),1)).T
    for d in data:
        d_add = np.array([list(d['x'].values())])
        X = np.append(X,d_add,axis=0)
    X = X[1:,:]

    x_data = []
    y_data = []
    i = 0 
    t_data = []
    while True:
        fid_data = np.zeros((1,len(X[0,:-1])))
        fid_sc = np.zeros((1,1))
        fid_t = []
        if i == len(X)-2:
            fid_data = np.append(fid_data,[X[i,:-1]],axis=0)
            fid_sc = np.append(fid_sc,[[data[i]['N']]],axis=0)
            fid_t.append(data[i]['t'])
            fid_t.append(data[i+1]['t'])
            fid_data = np.append(fid_data,[X[i+1,:-1]],axis=0)
            fid_sc = np.append(fid_sc,[[data[i+1]['N']]],axis=0)
            fid_data = fid_data[1:,:]
            fid_sc = fid_sc[1:,:]
            x_data.append(fid_data)
            t_data.append(fid_t)
            y_data.append(fid_sc)
            break 
        while X[i,-1] == X[i+1,-1]:
            fid_data = np.append(fid_data,[X[i,:-1]],axis=0)
            fid_sc = np.append(fid_sc,[[data[i]['N']]],axis=0)
            fid_t.append(data[i]['t'])
            i += 1
        fid_data = np.append(fid_data,[X[i,:-1]],axis=0)
        fid_sc = np.append(fid_sc,[[data[i]['N']]],axis=0)
        fid_t.append(data[i]['t'])
        i += 1
        fid_data = fid_data[1:,:]
        fid_sc = fid_sc[1:,:]
        x_data.append(fid_data)
        y_data.append(fid_sc)
        t_data.append(fid_t)


    z_data = [x_data[0]]
    for i in range(1,len(x_data)):
        new = np.concatenate((x_data[i-1],y_data[i-1]),axis=1)
        z_data.append(new)

    fig,ax=plt.subplots(1,1,figsize=(6,4))
    fid_v = np.linspace(0,1,5)
    j = 0 
    mean_p = []
    for i in range(len(t_data)):
        x_ax = [fid_v[j] for i in range(len(t_data[i]))]
        if i == 0:
            label1 = 'Initial Data'
            label2 = 'Mean'
        else:
            label1 = None
            label2 = None

        ax.scatter(x_ax,t_data[i],c='k',marker='+',label=label1)
        ax.scatter(fid_v[j],np.mean(t_data[i]),c='r',marker='+',label=label2)
        mean_p.append(np.mean(t_data[i]))
        j += 1
    ax.plot(fid_v,mean_p,ls='dashed',c='r',lw=1)
    ax.set_xlabel('Fidelity')
    ax.set_ylabel('Time (s)')
    ax.set_xticks(fid_v,fid_v)
    ax.legend(frameon=False)
    plt.savefig('outputs/initial_mf_points/time.png')

    return x_data,y_data,z_data


x_data,y_data,z_data = parse_initial_conditions('outputs/initial_mf_points/dataset_x.pickle')


# dgp = make_dgpMF_model(x_data, y_data, z_data)
# print('Running Optimisation!')
# run_adam(dgp,0.01, 5000)
# print('Done training')
