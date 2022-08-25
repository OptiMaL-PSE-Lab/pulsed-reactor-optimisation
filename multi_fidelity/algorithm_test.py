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


def f(x,fi):
        fi = 1-fi
        return (-(2*(x)**(2)+fi*4*x**3)-fi)/2#+np.random.uniform(0,0.5*fi,len(x))+3)/2

x_b = [-1,0.75]
n_eval = 100
fidels = [0,0.25,0.5,0.75,1]
lam = [1,2,4,6,10]
zeta = [4,5,4,3]

x = np.linspace(x_b[0],x_b[1],1000)

fig,axs = plt.subplots(2,1,figsize=(6,9))
for i in range(len(fidels)):
        y = f(x,fidels[i])
        #y = (y-np.mean(y))/np.std(y)
        axs[0].plot(x,y,c='k',alpha=0.5,ls=':')


n_init = 25
num = [10,5,5,3,2]
c = ['k','r','b','g','tab:orange']
lab = ['0','0.25','0.5','0.75','1']

x_data = []
y_data = []
for i in range(len(fidels)):
    x = np.linspace(x_b[0],x_b[1], num[i])[:,None]
    y = f(x.T[0],fidels[i])
    #y = (y-np.mean(y))/np.std(y)
    axs[0].scatter(x.T,y,s=5,c=c[i])
    x_data.append(x)
    y_data.append(np.array([y]).T)
#plt.savefig('multi_fidelity/test.png')

### Convert lists of arrays to ND-arrays augmented with fidelity indicators

X_train, Y_train = convert_xy_lists_to_arrays(x_data, y_data)

z_data = [x_data[i]]
for i in range(1,len(fidels)):
    new = np.concatenate((x_data[i-1],y_data[i-1]),axis=1)
    z_data.append(new)


dgp = make_dgpMF_model(x_data, y_data, z_data)


print('Running Optimisation!')
run_adam(dgp,0.01, 1000)
print('Done training')

n_plot = 200
plot = np.linspace(x_b[0],x_b[1],n_plot)[:,None]


something_all,mean_all,var_all = dgp.predict_all_layers(plot,400)
plot = plot.flatten()
for n in range(len(fidels)):
    
    mean = np.mean(mean_all[n],axis=0)[:,0]
    std  = np.mean(var_all[n],axis=0) 
    std = (np.sqrt(std) * 1.96)[:,0]
    B = 1
    axs[0].plot(plot,mean,c=c[n])
    axs[0].fill_between(plot, (mean - std), (mean + std), color=c[n], alpha=0.1,label=lab[n])
    axs[1].plot(plot,mean+B*std,c=c[n],label=lab[n])
    axs[1].legend()
    axs[0].set_title('Data and Estimation')
    axs[1].set_title('Aq Func')
    #axs[1].legend()

    axs[0].legend()




plt.savefig('multi_fidelity/algorithm_test.png',dpi=600)


