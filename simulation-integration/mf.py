from ast import Mult
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
from scipy.optimize import minimize
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
import matplotlib.pyplot as plt 
import numpy as np
from emukit.examples.multi_fidelity_dgp.multi_fidelity_deep_gp import DGP_Base, init_layers_mf,MultiFidelityDeepGP
from gpflow.kernels import RBF, White, Linear
from gpflow.likelihoods import Gaussian
from gpflow.actions import Loop, Action
from gpflow.mean_functions import Zero
from gpflow.training import AdamOptimizer
import gpflow.training.monitor as mon
from gpflow.training import AdamOptimizer, ScipyOptimizer, NatGradOptimizer
import time 

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

def eval_cfd(a, f, re, coil_rad, pitch, inversion_loc,fid):
    tube_rad = 0.0025
    length = 0.0753
    identifier = identifier = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print('Starting to mesh '+identifier)
    newcase = "outputs/geom/" + identifier
    create_mesh(coil_rad, tube_rad, pitch, length, inversion_loc, [fid,fid],path=newcase,validation=True,build=True)
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
        fid_t = np.zeros((1,1))
        if i == len(X)-2:
            fid_data = np.append(fid_data,[X[i,:-1]],axis=0)
            fid_sc = np.append(fid_sc,[[data[i]['N']]],axis=0)
            fid_t = np.append(fid_t,[[data[i]['t']]],axis=0)
            fid_t = np.append(fid_t,[[data[i+1]['t']]],axis=0)
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
            fid_t = np.append(fid_t,[[data[i]['t']]],axis=0)
            i += 1
        fid_data = np.append(fid_data,[X[i,:-1]],axis=0)
        fid_sc = np.append(fid_sc,[[data[i]['N']]],axis=0)
        fid_t = np.append(fid_t,[[data[i]['t']]],axis=0)
        i += 1
        fid_data = fid_data[1:,:]
        fid_sc = fid_sc[1:,:]
        fid_t = fid_t[1:,:]
        x_data.append(fid_data)
        y_data.append(fid_sc)
        t_data.append(fid_t)


    fig,ax=plt.subplots(1,1,figsize=(6,4))
    fid_v = np.linspace(0,1,5)
    j = 0 
    mean_t = []
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
        mean_t.append(np.mean(t_data[i]))
        j += 1
    ax.plot(fid_v,mean_t,ls='dashed',c='r',lw=1)
    ax.set_xlabel('Fidelity')
    ax.set_ylabel('Time (s)')
    ax.set_xticks(fid_v,fid_v)
    ax.legend(frameon=False)
    plt.savefig('outputs/initial_mf_points/time.pdf')

    return x_data,y_data,mean_t

try:
    with open("outputs/mf/logs.json", "r") as fp:
        data = json.load(fp) 

    def get_ind(f):
        fids = [0,0.25,0.5,0.75,1]
        for i in range(len(fids)):
            if f == fids[i]:
                return i

    x_data = [np.array([[0 for i in range(6)]]) for i in range(5)]
    y_data = [np.array([[0]]) for i in range(5)]
    for d in data.values():
        f_i = get_ind(d['fid'])
        x_data[f_i] = np.append(x_data[f_i],[list(d['params'].values())],axis=0)
        y_data[f_i] = np.append(y_data[f_i],[[d['target']]],axis=0)

    for i in range(len(x_data)):
        x_data[i] = x_data[i][1:,:]
        y_data[i] = y_data[i][1:,:]

    c = len(data)

    x_data_og,y_data_og,t_data = parse_initial_conditions('outputs/initial_mf_points/dataset_x.pickle')
except: 
    x_data,y_data,t_data = parse_initial_conditions('outputs/initial_mf_points/dataset_x.pickle')
    fid_list = [0,0.25,0.5,0.75,1]
    data = {}
    keys = ['a','f','re','coil_rad','pitch','inversion_loc']
    c = 1 
    for i in range(len(x_data)):
        for j in range(len(x_data[i])):
            p = {}
            p['target'] = y_data[i][j,0]
            pa = {}
            for k in range(len(keys)):
                pa[keys[k]] = x_data[i][j,k]
            p['params'] = pa
            p['fid'] = fid_list[i]
            data[c] = p
            c += 1

    with open("outputs/mf/logs.json", "w") as fp:
        json.dump(data,fp) 

while True:
    dgp = MultiFidelityDeepGP(x_data,y_data,n_iter=40000,multi_step_training=False)
    print('Optimizing')

    t_data = np.array(t_data)
    t_data = 1/((t_data)/(max(t_data)))

    #dgp.optimize()

    def aq_fun(x,dgp):
        s = time.time()
        res_s,res_mean,res_var = dgp.predict_all(np.array([x]))
        for i in range(len(res_mean)):
            res_mean[i] = np.mean(res_mean[i],axis=0)
            res_var[i] = np.mean(res_var[i],axis=0)
        return -(res_mean[-1] + 2.5*res_var[-1])

    fid_list = [0,0.25,0.5,0.75,1]
    lb = np.array([0.001,2,10,0.0075,0.005,0])
    ub = np.array([0.008,8,50,0.015,0.0125,1])
    n_multi = 5
    x0_list = np.random.uniform(0,1,(len(lb),n_multi))
    x0_list = np.array([[(x0_list[i,j]) * (ub[i]-lb[i]) + lb[i] for i in range(len(lb))]for j in range(n_multi)])

    f_max = -1E20
    i = 1 
    for x0 in x0_list:
        print("Multistart: ",i,' Best upper-bound: ',f_max)
        i += 1
        res = minimize(aq_fun,x0,args=(dgp),method='Nelder-Mead',options={'maxiter':1000})
        if res.fun > f_max:
            f_max = res.fun
            res_best = res


    xn = [0,0,0,0,0,0] 
    def fid_choice(x,dgp):
        res_s,res_mean,res_var = dgp.predict_all(np.array([x]))
        for i in range(len(res_mean)):
            res_mean[i] = np.mean(res_mean[i],axis=0)
            res_var[i] = np.mean(res_var[i],axis=0)    

        choices = (np.array(res_var) * 2.5) 
        choices = np.array([choices[i][0,0] for i in range(len(choices))])
        choices *= t_data
        ind = np.argmax(choices)
        print('CHOICES FOR FIDELITY')
        print(choices)
        return fid_list[ind],ind

    xn = res_best.x
    fid,f_ind = fid_choice(xn,dgp)
    N = eval_cfd(xn[0],xn[1],xn[2],xn[3],xn[4],xn[5],fid)
    x_data[f_ind] = np.append(x_data[f_ind],[xn],axis=0)
    y_data[f_ind] = np.append(y_data[f_ind],[[N]],axis=0)
    keys = ['a','f','re','coil_rad','pitch','inversion_loc'] 
    p = {}
    p['target'] = N
    pa = {}
    for k in range(len(keys)):
        pa[keys[k]] = xn[k]
    p['params'] = pa
    p['fid'] = fid
    data[c] = p
    c += 1

    with open("outputs/mf/logs.json", "w") as fp:
        json.dump(data,fp) 


