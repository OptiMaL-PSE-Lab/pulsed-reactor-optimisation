from bayes_opt_with_constraints.bayes_opt import BayesianOptimization, UtilityFunction
from utils import  newJSONLogger
from bayes_opt_with_constraints.bayes_opt.event import Events
import json
from datetime import datetime
import os 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import gridspec
from distutils.dir_util import copy_tree


def eval_cfd_a(a):
    f = 5
    re = 50 
    identifier = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
    print('Starting to copy mmulation-integration/outputesh '+identifier)
    newcase = "outputs/a_only/" + identifier
    os.mkdir(newcase)
    copy_tree("mesh_generation/experimental_mesh", newcase)   
    print('Starting simulation...')
    vel = vel_calc(re)
    parse_conditions(newcase, a, f, vel)
    time, value = run_cfd(newcase)
    N = calculate_N(value, time,newcase)
    return N

logger = newJSONLogger(path="outputs/a_only/logs.json")

# defining utility function
utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

# def eval_cfd_a(a):
#     b = a * 1000
#     return np.exp(-(b - 3)**2) + np.exp(-(b - 6)**2/10) + 1/ (b**2 + 1)

optimizer = BayesianOptimization(
    f=eval_cfd_a,
    pbounds={
        "a": (0.001, 0.008),
    },
    pcons=[],
    verbose=2,
    random_state=1
)


def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x,iteration):
    fig = plt.figure(figsize=(7, 4))
    plt.subplots_adjust(left=0.1,right=0.95,top=0.95)
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    axis.set_xticks([],[])
    
    x_obs = np.array([[res["params"]["a"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    
    mu, sigma = posterior(optimizer, x_obs, y_obs, x)

    axis.scatter(x_obs.flatten(), y_obs, marker='+',s=80,lw=1, label=u'Observations', color='k')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.1, fc='k', ec='None',label='95% confidence interval')
    
    axis.set_xlim((0.001,0.008))
    axis.set_ylim((None, None))
    axis.set_ylabel('N')
    
    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='k')
    acq.scatter(x[np.argmax(utility)], np.max(utility), marker='+', s=80, c='k',lw=1,
             label=u'Next Best Guess')
    acq.set_xlim((0.001, 0.008))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('UCB')
    acq.set_xlabel('a')
    
    axis.legend(frameon=False)
    acq.legend(frameon=False)
    fig.savefig('outputs/a_only/bo_iteration_'+str(iteration)+'.png')
    return 


# setting up the optimisation problem

# Opening JSON file
try:
    logs = []
    with open("outputs/a_only/logs.json") as f:
        for line in f:
            logs.append(json.loads(line))

    for log in logs:
        optimizer.register(params=log["params"], target=log["target"])
except FileNotFoundError:
    pass

# assign logger to optimizer
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
x = np.linspace(0.001, 0.008, 500).reshape(-1, 1)
iteration = 0
n_init = 5
init_points = np.linspace(0.001,0.008,n_init).reshape(-1,1)
keys = ['a']
for p in init_points:
    p_dict = {}
    for i in range(len(keys)):
        p_dict[keys[i]] = p[i]
    target = eval_cfd_a(**p_dict)
    optimizer.register(params=p_dict, target=target)
    plot_gp(optimizer,x,iteration)

while True:
    next_point = optimizer.suggest(utility)
    target = eval_cfd_a(**next_point)
    iteration += 1 
    optimizer.register(params=next_point, target=target)
    plot_gp(optimizer,x,iteration)
