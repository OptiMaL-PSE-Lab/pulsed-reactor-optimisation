from bayes_opt_with_constraints.bayes_opt import BayesianOptimization, UtilityFunction
from utils import  newJSONLogger
from bayes_opt_with_constraints.bayes_opt.event import Events
import json
from datetime import datetime
import os
import shutil  
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import gridspec
from distutils.dir_util import copy_tree
from utils import vel_calc,parse_conditions,run_cfd,calculate_N

def eval_cfd_a(a,f):
    re = 50 
    identifier = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print('Starting to copy mesh')
    newcase = "outputs/a_and_f/" + identifier
    os.mkdir(newcase)
    copy_tree("mesh_generation/experimental_mesh", newcase)   
    print('Starting simulation...')
    vel = vel_calc(re)
    parse_conditions(newcase, a, f, vel)
    time, value = run_cfd(newcase)
    N = calculate_N(value, time,newcase)
    return N

logger = newJSONLogger(path="outputs/a_and_f/logs.json")

# defining utility function

utility_f = UtilityFunction(kind="ucb", kappa=5, xi=0.0)
# def eval_cfd_a(a):
#     b = a * 1000
#     return np.exp(-(b - 3)**2) + np.exp(-(b - 6)**2/10) + 1/ (b**2 + 1)

optimizer = BayesianOptimization(
    f=eval_cfd_a,
    pbounds={
        "a": (0.001, 0.008),
        "f": (2,8),
    },
    pcons=[],
    verbose=2,
    random_state=1
)


def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


# Opening JSON file
try:
    logs = []
    with open("outputs/a_and_f/logs.json") as fi:
        print(fi)
        for line in fi:
            print(line)
            logs.append(json.loads(line))
    print(logs)
    for log in logs:
        optimizer.register(params=log["params"], target=log["target"])
        print(optimizer)
except FileNotFoundError:
    print('FILE NOT FOUND')
    pass

# assign logger to optimizer
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
n_init = 5
a_init = np.linspace(0.001,0.008,n_init).reshape(-1,1)
f_init = np.linspace(2,8,n_init).reshape(-1,1)
np.random.shuffle(f_init)
init_points = np.concatenate((a_init,f_init),axis=1)

keys = ['a','f']
for p in init_points:
    p_dict = {}
    for i in range(len(keys)):
        p_dict[keys[i]] = p[i]
    target = eval_cfd_a(**p_dict)
    optimizer.register(params=p_dict, target=target)

while True:
    next_point = optimizer.suggest(utility_f)
    target = eval_cfd_a(**next_point)
    optimizer.register(params=next_point, target=target)
