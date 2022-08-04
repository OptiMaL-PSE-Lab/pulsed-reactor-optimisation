import GPy
import emukit.test_functions
import numpy as np 
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
import matplotlib.pyplot as plt 
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, NonLinearMultiFidelityModel


def f(x,u):
    return (x-0.5)**2 - 2*u


fid = [0.1,0.05,0.01]
num = [20,8,4]
c = ['k','r','b']
lab = ['low','medium','high']
plt.figure()
x_data = []
y_data = []
for i in range(len(fid)):
    x = np.linspace(0, 1, num[i])[:,None]
    y = f(x.T[0],fid[i])
    if len(fid) == 3:
        plt.scatter(x.T,y,s=5,c=c[i])
    else:
        plt.scatter(x.T,y,s=5)
    x_data.append(x)
    y_data.append(np.array([y]).T)
#plt.savefig('multi_fidelity/test.png')

### Convert lists of arrays to ND-arrays augmented with fidelity indicators
X_train, Y_train = convert_xy_lists_to_arrays(x_data, y_data)


base_kernel = GPy.kern.RBF
k = make_non_linear_kernels(base_kernel, len(fid), X_train.shape[1] - 1)
m = NonLinearMultiFidelityModel(X_train, Y_train, n_fidelities=len(fid), kernels=k, 
                                verbose=True, optimization_restarts=10)
for m in m.models:
    m.Gaussian_noise.variance.fix(0)
    
m.optimize()

n_plot = 200
plot = np.linspace(0,1,n_plot)[:,None]
X_plot = convert_x_list_to_array([plot for i in range(len(fid))])
for n in range(len(fid)):
    X_plot_t = X_plot[n*n_plot:n_plot+n*n_plot]

    ## Compute mean and variance predictions

    mean, var = m.predict(X_plot_t)
    std = np.sqrt(var)
    if len(fid) == 3:
        plt.fill_between(plot.flatten(), (mean - 1.96*std).flatten(), 
                        (mean + 1.96*std).flatten(), color=c[n], alpha=0.1,label=lab[n])
        plt.plot(plot, mean, '--', color=c[n])

    else:
        plt.fill_between(plot.flatten(), (mean - 1.96*std).flatten(), 
                        (mean + 1.96*std).flatten(), alpha=0.1)
        plt.plot(plot, mean, '--')

    

if len(fid)==3:
    plt.legend()
plt.savefig('multi_fidelity/one_dim_example.png')


